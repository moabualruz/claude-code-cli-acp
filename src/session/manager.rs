use std::{
    future::Future,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use agent_client_protocol::schema::{
    McpServer, PromptRequest, SessionConfigOption, SessionConfigValueId, SessionId,
    SessionModeState, SessionModelState, SessionUpdate,
};

use crate::{
    compat::claude_probe::ClaudeCli,
    config::{
        session::SessionConfigState,
        settings::{SettingsPaths, load_merged_settings},
    },
    pty::session::{ClaudePtyConfig, ClaudePtySession},
    terminal::recognizers::{self, PermissionDecision, PermissionDialog},
    transcript::{
        events::{TranscriptEvent, TranscriptEventKind},
        tailer::{TranscriptLocator, TranscriptTailer},
    },
};

const CLAUDE_ASSISTANT_LINE_MARKER: char = '⏺';

#[derive(Clone)]
pub struct SessionManager {
    claude: ClaudeCli,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            claude: ClaudeCli::from_env(),
        }
    }

    pub fn create_session(
        &self,
        session_id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> anyhow::Result<std::sync::Arc<ManagedSession>> {
        Ok(std::sync::Arc::new(ManagedSession::new(
            self.claude.clone(),
            session_id,
            cwd,
            mcp_servers,
            None,
        )))
    }

    pub fn load_session(
        &self,
        session_id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> anyhow::Result<std::sync::Arc<ManagedSession>> {
        self.create_session(session_id, cwd, mcp_servers)
    }

    pub fn create_print_session(
        &self,
        session_id: String,
        cwd: PathBuf,
        model: Option<String>,
    ) -> anyhow::Result<ManagedSession> {
        Ok(ManagedSession::new(
            self.claude.clone(),
            SessionId::new(session_id),
            cwd,
            Vec::new(),
            model,
        ))
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TurnOptions {
    pub timeout: Duration,
    pub model: Option<String>,
    pub permission_mode: Option<String>,
    pub resume: Option<String>,
    pub continue_last: bool,
    pub initial_prompt_argument: bool,
    pub attach_on_timeout: bool,
    pub attach_on_permission: bool,
}

impl TurnOptions {
    pub fn from_prompt_request(_request: &PromptRequest) -> Self {
        Self {
            timeout: Duration::from_secs(120),
            model: None,
            permission_mode: None,
            resume: None,
            continue_last: false,
            initial_prompt_argument: false,
            attach_on_timeout: false,
            attach_on_permission: false,
        }
    }
}

pub struct ManagedSession {
    claude: ClaudeCli,
    session_id: SessionId,
    cwd: PathBuf,
    mcp_servers: Vec<McpServer>,
    model: Mutex<Option<String>>,
    permission_mode: Mutex<Option<String>>,
    config: Mutex<SessionConfigState>,
    pty: Mutex<Option<ClaudePtySession>>,
    prompt_lock: tokio::sync::Mutex<()>,
}

impl ManagedSession {
    fn new(
        claude: ClaudeCli,
        session_id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        model: Option<String>,
    ) -> Self {
        let settings = SettingsPaths::for_cwd(&cwd)
            .map(|paths| load_merged_settings(&paths).settings)
            .unwrap_or_default();
        let mut config = SessionConfigState::from_settings(&settings);
        if let Some(model) = model.as_deref()
            && let Ok(resolved) = config.set_model(model)
        {
            drop(resolved);
        }
        Self {
            claude,
            session_id,
            cwd,
            mcp_servers,
            model: Mutex::new(model),
            permission_mode: Mutex::new(None),
            config: Mutex::new(config),
            pty: Mutex::new(None),
            prompt_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn set_model(&self, model: Option<String>) -> anyhow::Result<()> {
        if let Some(model) = model.as_deref() {
            let resolved = self.config.lock().unwrap().set_model(model)?;
            *self.model.lock().unwrap() = Some(resolved);
        } else {
            *self.model.lock().unwrap() = None;
        }
        Ok(())
    }

    pub fn set_permission_mode(&self, permission_mode: Option<String>) -> anyhow::Result<()> {
        if let Some(permission_mode) = permission_mode.as_deref() {
            self.config.lock().unwrap().set_mode(permission_mode)?;
            *self.permission_mode.lock().unwrap() =
                Some(self.config.lock().unwrap().mode().to_string());
        } else {
            *self.permission_mode.lock().unwrap() = None;
        }
        Ok(())
    }

    pub fn modes(&self) -> SessionModeState {
        self.config.lock().unwrap().modes()
    }

    pub fn models(&self) -> SessionModelState {
        self.config.lock().unwrap().models()
    }

    pub fn config_options(&self) -> Vec<SessionConfigOption> {
        self.config.lock().unwrap().config_options()
    }

    pub fn set_config_option(
        &self,
        config_id: &str,
        value: &SessionConfigValueId,
    ) -> anyhow::Result<Option<SessionUpdate>> {
        let update = self.config.lock().unwrap().set_option(config_id, value)?;
        match config_id {
            "mode" => {
                *self.permission_mode.lock().unwrap() =
                    Some(self.config.lock().unwrap().mode().to_string());
            }
            "model" => {
                *self.model.lock().unwrap() = Some(self.config.lock().unwrap().model().to_string());
            }
            _ => {}
        }
        Ok(update)
    }

    pub async fn prompt(&self, prompt: String, options: TurnOptions) -> anyhow::Result<TurnOutput> {
        self.prompt_with_permission_handler(prompt, options, |request| async move {
            anyhow::bail!(
                "Claude requested permission before transcript completion for session {}: {}",
                request.session_id,
                request.dialog.title
            )
        })
        .await
    }

    pub async fn prompt_with_permission_handler<F, Fut>(
        &self,
        prompt: String,
        options: TurnOptions,
        mut permission_handler: F,
    ) -> anyhow::Result<TurnOutput>
    where
        F: FnMut(PendingPermission) -> Fut + Send,
        Fut: Future<Output = anyhow::Result<PermissionDecision>> + Send,
    {
        let _prompt_guard = self.prompt_lock.lock().await;
        let (mut pty, reused_pty) = self.ensure_pty(&options, &prompt)?;
        let locator = TranscriptLocator::default_home()?;
        let mut tailer =
            TranscriptTailer::from_locator_at_end(self.session_id.0.to_string(), &locator)?;
        if !options.initial_prompt_argument || reused_pty {
            wait_for_idle_prompt(&mut pty, options.timeout)?;
            pty.submit_prompt(&prompt)?;
        }

        let deadline = tokio::time::Instant::now() + options.timeout;
        let mut events = Vec::new();
        let mut active_permission_fingerprint: Option<String> = None;
        loop {
            if tailer.is_none() {
                tailer = TranscriptTailer::from_locator(self.session_id.0.to_string(), &locator)?;
            }
            if let Some(tailer) = tailer.as_mut() {
                events.extend(tailer.poll()?);
            }
            let screen_text = pty.screen_snapshot();
            let screen_is_idle = screen_text
                .as_deref()
                .ok()
                .is_some_and(recognizers::recognize_idle);
            let has_assistant_event = events.iter().any(is_assistant_terminal_event);
            if has_assistant_event && (screen_is_idle || screen_text.is_err()) {
                break;
            }
            if !has_assistant_event
                && screen_is_idle
                && screen_text
                    .as_deref()
                    .ok()
                    .and_then(assistant_text_from_screen)
                    .is_some()
            {
                events.clear();
                break;
            }
            if let Some(dialog) = pty.permission_dialog()? {
                let fingerprint = permission_fingerprint(&dialog);
                if active_permission_fingerprint.as_deref() == Some(&fingerprint) {
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    continue;
                }
                if options.attach_on_permission {
                    pty.detach_for_user()?;
                    anyhow::bail!(
                        "attached user to Claude session {} for permission request",
                        self.session_id.0
                    );
                }
                let decision = permission_handler(PendingPermission {
                    session_id: self.session_id.clone(),
                    dialog,
                })
                .await?;
                if !pty.select_permission(decision)? {
                    anyhow::bail!(
                        "unable to select Claude permission option {:?} for session {}",
                        decision,
                        self.session_id.0
                    );
                }
                active_permission_fingerprint = Some(fingerprint);
            } else {
                active_permission_fingerprint = None;
            }
            if tokio::time::Instant::now() >= deadline {
                if options.attach_on_timeout {
                    pty.detach_for_user()?;
                }
                let screen_status = pty
                    .screen_snapshot()
                    .map(|text| recognizers::recognize_screen(&text))
                    .unwrap_or(recognizers::ScreenStatus::Unknown);
                anyhow::bail!(
                    "timed out waiting for Claude transcript completion for session {} (screen status: {:?})",
                    self.session_id.0,
                    screen_status
                );
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        let screen_text = pty
            .screen_snapshot()
            .ok()
            .map(|text| assistant_text_from_screen(&text).unwrap_or(text));
        *self.pty.lock().unwrap() = Some(pty);
        Ok(TurnOutput {
            events,
            screen_text,
        })
    }

    pub async fn cancel(&self) -> anyhow::Result<()> {
        if let Some(pty) = self.pty.lock().unwrap().as_mut() {
            pty.send_interrupt()?;
        }
        Ok(())
    }

    pub async fn shutdown(&self) -> anyhow::Result<()> {
        if let Some(mut pty) = self.pty.lock().unwrap().take() {
            pty.send_exit()?;
            pty.terminate()?;
        }
        Ok(())
    }

    fn ensure_pty(
        &self,
        options: &TurnOptions,
        prompt: &str,
    ) -> anyhow::Result<(ClaudePtySession, bool)> {
        if let Some(pty) = self.pty.lock().unwrap().take() {
            return Ok((pty, true));
        }
        let mut model = self
            .model
            .lock()
            .unwrap()
            .clone()
            .or_else(|| Some(self.config.lock().unwrap().model().to_string()));
        if model.as_deref() == Some("default") {
            model = None;
        }
        if options.model.is_some() {
            model = options.model.clone();
        }
        let permission_mode = options
            .permission_mode
            .clone()
            .or_else(|| self.permission_mode.lock().unwrap().clone());
        let config = ClaudePtyConfig {
            executable: self.claude.executable().to_path_buf(),
            cwd: self.cwd.clone(),
            session_id: self.session_id.0.to_string(),
            model,
            permission_mode,
            setting_sources: std::env::var("CLAUDE_CODE_ACP_SETTING_SOURCES")
                .ok()
                .filter(|sources| !sources.trim().is_empty()),
            resume: options.resume.clone(),
            continue_last: options.continue_last,
            mcp_servers: self.mcp_servers.clone(),
            extra_args: if options.initial_prompt_argument {
                vec![prompt.into()]
            } else {
                Vec::new()
            },
            rows: 24,
            cols: 80,
        };
        Ok((ClaudePtySession::spawn(config)?, false))
    }
}

#[derive(Clone, Debug)]
pub struct PendingPermission {
    pub session_id: SessionId,
    pub dialog: PermissionDialog,
}

fn is_assistant_terminal_event(event: &TranscriptEvent) -> bool {
    match event.kind {
        TranscriptEventKind::AssistantMessage => event
            .text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty()),
        TranscriptEventKind::ToolResult => {
            event.session_id.is_some()
                && event
                    .text
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())
        }
        _ => false,
    }
}

fn assistant_text_from_screen(text: &str) -> Option<String> {
    text.lines().rev().find_map(assistant_text_from_line)
}

fn assistant_text_from_line(line: &str) -> Option<String> {
    let after_marker = line
        .trim_start()
        .strip_prefix(CLAUDE_ASSISTANT_LINE_MARKER)?
        .strip_prefix(' ')?;
    let text = strip_duration_suffix(after_marker.trim()).trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn strip_duration_suffix(text: &str) -> &str {
    text.strip_suffix(')')
        .and_then(|without_close| without_close.rsplit_once(" ("))
        .filter(|(_, suffix)| {
            suffix.ends_with('s')
                && suffix[..suffix.len() - 1]
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.')
        })
        .map(|(prefix, _)| prefix)
        .unwrap_or(text)
}

#[cfg(test)]
mod tests {
    use super::assistant_text_from_screen;

    #[test]
    fn assistant_text_from_screen_requires_assistant_line_shape() {
        let screen = "user typed a marker: ⏺ not assistant\n  ⏺ OK. (1.0s)\n";
        assert_eq!(assistant_text_from_screen(screen).as_deref(), Some("OK."));
    }

    #[test]
    fn assistant_text_from_screen_uses_latest_assistant_line() {
        let screen = "  ⏺ stale text\n  ⏺ current text\n";
        assert_eq!(
            assistant_text_from_screen(screen).as_deref(),
            Some("current text")
        );
    }

    #[test]
    fn assistant_text_from_screen_preserves_non_duration_parenthetical() {
        let screen = "  ⏺ use foo (bar) now\n";
        assert_eq!(
            assistant_text_from_screen(screen).as_deref(),
            Some("use foo (bar) now")
        );
    }

    #[test]
    fn assistant_text_from_screen_rejects_marker_inside_non_assistant_line() {
        let screen = "tool output mentions ⏺ but is not an assistant line\n";
        assert_eq!(assistant_text_from_screen(screen), None);
    }

    #[test]
    fn assistant_text_from_screen_rejects_empty_assistant_line() {
        assert_eq!(assistant_text_from_screen("⏺   \n"), None);
    }
}

fn permission_fingerprint(dialog: &PermissionDialog) -> String {
    format!(
        "{}::{:?}",
        dialog.title,
        dialog
            .options
            .iter()
            .map(|option| (&option.accelerator, &option.label, option.decision))
            .collect::<Vec<_>>()
    )
}

fn wait_for_idle_prompt(pty: &mut ClaudePtySession, timeout: Duration) -> anyhow::Result<()> {
    let startup_timeout = timeout.min(Duration::from_secs(20));
    let deadline = std::time::Instant::now() + startup_timeout;
    let mut confirmed_workspace_trust = false;
    loop {
        let screen_status = pty
            .screen_snapshot()
            .map(|text| recognizers::recognize_screen(&text))
            .unwrap_or(recognizers::ScreenStatus::Unknown);
        match screen_status {
            recognizers::ScreenStatus::Idle => return Ok(()),
            recognizers::ScreenStatus::WorkspaceTrust if !confirmed_workspace_trust => {
                pty.write_bytes(b"\r")?;
                confirmed_workspace_trust = true;
            }
            _ => {}
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for Claude interactive prompt");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

pub struct TurnOutput {
    pub events: Vec<TranscriptEvent>,
    pub screen_text: Option<String>,
}

impl TurnOutput {
    pub fn final_text(&self) -> String {
        self.events
            .iter()
            .filter(|event| matches!(event.kind, TranscriptEventKind::AssistantMessage))
            .filter_map(|event| event.text.as_deref())
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string()
            .or_else_screen(self.screen_text.as_deref())
    }

    pub fn model(&self) -> Option<String> {
        self.events.iter().find_map(|event| event.model.clone())
    }
}

trait ScreenFallback {
    fn or_else_screen(self, screen: Option<&str>) -> String;
}

impl ScreenFallback for String {
    fn or_else_screen(self, screen: Option<&str>) -> String {
        if self.is_empty() {
            screen.unwrap_or_default().to_string()
        } else {
            self
        }
    }
}
