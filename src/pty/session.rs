use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use agent_client_protocol::schema::McpServer;
use anyhow::{Context, anyhow};
use portable_pty::{Child, CommandBuilder, PtySize, native_pty_system};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::pty::input;
use crate::terminal::{
    recognizers::{PermissionDecision, PermissionDialog, recognize_permission_dialog},
    screen::TerminalScreen,
};

// Claude Code 2.1.169 paints the TUI prompt editor lazily after startup.
const CLAUDE_2_1_169_TUI_SETTLE_DELAY: Duration = Duration::from_millis(2_000);
// Claude Code 2.1.169 can buffer typed prompt bytes before Enter is accepted.
const CLAUDE_2_1_169_PRE_ENTER_DELAY: Duration = Duration::from_millis(800);
// Claude Code 2.1.169 sometimes leaves echoed prompt text idle; retry Enter then.
const CLAUDE_2_1_169_SUBMIT_RETRY_DELAYS: [Duration; 3] = [
    Duration::from_millis(1_000),
    Duration::from_millis(1_500),
    Duration::from_millis(2_000),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaudePtyConfig {
    pub executable: PathBuf,
    pub cwd: PathBuf,
    pub session_id: String,
    pub model: Option<String>,
    pub permission_mode: Option<String>,
    pub setting_sources: Option<String>,
    pub resume: Option<String>,
    pub continue_last: bool,
    pub mcp_servers: Vec<McpServer>,
    pub extra_args: Vec<OsString>,
    pub rows: u16,
    pub cols: u16,
}

impl Default for ClaudePtyConfig {
    fn default() -> Self {
        Self {
            executable: PathBuf::from("claude"),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            session_id: uuid::Uuid::new_v4().to_string(),
            model: None,
            permission_mode: None,
            setting_sources: None,
            resume: None,
            continue_last: false,
            mcp_servers: Vec::new(),
            extra_args: Vec::new(),
            rows: 24,
            cols: 80,
        }
    }
}

impl ClaudePtyConfig {
    pub fn launch_argv(&self) -> anyhow::Result<Vec<OsString>> {
        let mut argv = Vec::new();
        argv.push(self.executable.clone().into_os_string());
        argv.push("--session-id".into());
        argv.push(self.session_id.clone().into());
        if let Some(model) = &self.model {
            argv.push("--model".into());
            argv.push(model.into());
        }
        if let Some(permission_mode) = &self.permission_mode {
            argv.push("--permission-mode".into());
            argv.push(permission_mode.into());
        }
        if let Some(setting_sources) = &self.setting_sources {
            argv.push("--setting-sources".into());
            argv.push(setting_sources.into());
        }
        if let Some(session) = &self.resume {
            argv.push("--resume".into());
            argv.push(session.into());
        }
        if self.continue_last {
            argv.push("--continue".into());
        }
        argv.extend(self.extra_args.iter().cloned());
        reject_print_mode_args(argv.iter().skip(1))?;
        Ok(argv)
    }
}

pub struct ClaudePtySession {
    child: Box<dyn Child + Send + Sync>,
    writer: Box<dyn Write + Send>,
    screen: Arc<Mutex<TerminalScreen>>,
    reader_thread: Option<JoinHandle<()>>,
    executable: PathBuf,
    cwd: PathBuf,
    session_id: String,
    model: Option<String>,
    permission_mode: Option<String>,
    generated_mcp_config: Option<PathBuf>,
}

impl ClaudePtySession {
    pub fn spawn(config: ClaudePtyConfig) -> anyhow::Result<Self> {
        let generated_mcp_config = write_mcp_config_file(&config)?;
        let mut argv = config.launch_argv()?;
        if let Some(path) = &generated_mcp_config {
            argv.push("--mcp-config".into());
            argv.push(path.into());
            argv.push("--strict-mcp-config".into());
        }
        let mut command = CommandBuilder::new(&argv[0]);
        command.args(argv.iter().skip(1));
        command.cwd(&config.cwd);
        command.env("CLAUDE_CODE_NO_FLICKER", "1");
        command.env("CLAUDE_CODE_DISABLE_MOUSE", "1");
        command.env("CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION", "false");

        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: config.rows,
                cols: config.cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("open pty")?;
        let child = pair
            .slave
            .spawn_command(command)
            .context("spawn claude pty")?;
        let mut reader = pair.master.try_clone_reader().context("clone pty reader")?;
        let writer = pair.master.take_writer().context("take pty writer")?;
        drop(pair.slave);

        let screen = Arc::new(Mutex::new(TerminalScreen::new(config.rows, config.cols)));
        let reader_screen = Arc::clone(&screen);
        let reader_thread = thread::spawn(move || {
            let mut buffer = [0_u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => {
                        if let Ok(mut screen) = reader_screen.lock() {
                            screen.process(&buffer[..read]);
                        } else {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            child,
            writer,
            screen,
            reader_thread: Some(reader_thread),
            executable: config.executable,
            cwd: config.cwd,
            session_id: config.session_id,
            model: config.model,
            permission_mode: config.permission_mode,
            generated_mcp_config,
        })
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        self.writer.write_all(bytes).context("write pty bytes")?;
        self.writer.flush().context("flush pty bytes")
    }

    pub fn submit_prompt(&mut self, prompt: &str) -> anyhow::Result<()> {
        thread::sleep(CLAUDE_2_1_169_TUI_SETTLE_DELAY);
        self.write_bytes(prompt.as_bytes())?;
        thread::sleep(CLAUDE_2_1_169_PRE_ENTER_DELAY);
        self.write_bytes(b"\r")?;
        self.retry_submit_if_prompt_still_idle(prompt)
    }

    fn retry_submit_if_prompt_still_idle(&mut self, prompt: &str) -> anyhow::Result<()> {
        let Some(needle) = prompt_echo_needle(prompt) else {
            return Ok(());
        };
        for delay in CLAUDE_2_1_169_SUBMIT_RETRY_DELAYS {
            thread::sleep(delay);
            let screen_text = self.screen_snapshot()?;
            if screen_text.contains(&needle)
                && crate::terminal::recognizers::recognize_screen(&screen_text)
                    == crate::terminal::recognizers::ScreenStatus::Idle
            {
                self.write_bytes(b"\r")?;
            } else {
                break;
            }
        }
        Ok(())
    }

    pub fn send_exit(&mut self) -> anyhow::Result<()> {
        self.write_bytes(&input::slash_command("exit"))
    }

    pub fn send_interrupt(&mut self) -> anyhow::Result<()> {
        self.write_bytes(&input::ctrl_c())
    }

    pub fn terminate(&mut self) -> anyhow::Result<()> {
        if self.child.try_wait().context("poll pty child")?.is_none() {
            self.child.kill().context("kill pty child")?;
        }
        if let Some(path) = self.generated_mcp_config.take() {
            drop(std::fs::remove_file(path));
        }
        Ok(())
    }

    pub fn screen_snapshot(&self) -> anyhow::Result<String> {
        Ok(self
            .screen
            .lock()
            .map(|screen| screen.text())
            .unwrap_or_else(|_| String::new()))
    }

    pub fn is_idle(&self) -> bool {
        self.screen_snapshot()
            .map(|text| crate::terminal::recognizers::recognize_idle(&text))
            .unwrap_or(false)
    }

    pub fn permission_dialog(&self) -> anyhow::Result<Option<PermissionDialog>> {
        Ok(recognize_permission_dialog(&self.screen_snapshot()?))
    }

    pub fn select_permission(&mut self, decision: PermissionDecision) -> anyhow::Result<bool> {
        let Some(dialog) = self.permission_dialog()? else {
            return Ok(false);
        };
        let Some(bytes) = input::permission_choice(&dialog, decision) else {
            return Ok(false);
        };
        self.write_bytes(&bytes)?;
        Ok(true)
    }

    pub fn detach_for_user(&mut self) -> anyhow::Result<()> {
        self.terminate()?;

        let mut command = std::process::Command::new(&self.executable);
        command
            .arg("--resume")
            .arg(&self.session_id)
            .current_dir(&self.cwd);
        if let Some(model) = &self.model {
            command.arg("--model").arg(model);
        }
        if let Some(permission_mode) = &self.permission_mode {
            command.arg("--permission-mode").arg(permission_mode);
        }
        let status = command
            .status()
            .context("attach user to resumed Claude session")?;
        if status.success() {
            Ok(())
        } else {
            Err(anyhow!(
                "attached Claude session exited with status {status}"
            ))
        }
    }
}

fn prompt_echo_needle(prompt: &str) -> Option<String> {
    let line = prompt.lines().rev().find(|line| !line.trim().is_empty())?;
    let trimmed = line.trim();
    let char_count = trimmed.chars().count();
    let start = char_count.saturating_sub(48);
    Some(trimmed.chars().skip(start).collect())
}

#[cfg(test)]
mod tests {
    use super::prompt_echo_needle;

    #[test]
    fn prompt_echo_needle_uses_last_non_empty_trimmed_line() {
        assert_eq!(
            prompt_echo_needle("first line\n\n  final answer  "),
            Some("final answer".to_string())
        );
    }

    #[test]
    fn prompt_echo_needle_truncates_to_last_48_chars() {
        let prompt = "1234567890".repeat(6);
        let expected: String = prompt.chars().skip(12).collect();
        assert_eq!(prompt_echo_needle(&prompt), Some(expected));
    }

    #[test]
    fn prompt_echo_needle_returns_none_for_blank_prompt() {
        assert_eq!(prompt_echo_needle(" \n\t\n"), None);
    }
}

fn write_mcp_config_file(config: &ClaudePtyConfig) -> anyhow::Result<Option<PathBuf>> {
    if config.mcp_servers.is_empty() {
        return Ok(None);
    }
    let path = std::env::temp_dir().join(format!(
        "claude-code-cli-acp-mcp-{}-{}.json",
        config.session_id,
        Uuid::new_v4()
    ));
    let body = serde_json::to_vec(&mcp_config_json(&config.mcp_servers))?;
    std::fs::write(&path, body).context("write temporary Claude MCP config")?;
    Ok(Some(path))
}

pub fn mcp_config_json(servers: &[McpServer]) -> Value {
    let mut mcp_servers = serde_json::Map::new();
    for server in servers {
        match server {
            McpServer::Stdio(server) => {
                let env = server
                    .env
                    .iter()
                    .map(|var| (var.name.clone(), Value::String(var.value.clone())))
                    .collect::<serde_json::Map<_, _>>();
                mcp_servers.insert(
                    server.name.clone(),
                    json!({
                        "type": "stdio",
                        "command": server.command,
                        "args": server.args,
                        "env": env,
                    }),
                );
            }
            McpServer::Http(server) => {
                mcp_servers.insert(
                    server.name.clone(),
                    json!({
                        "type": "http",
                        "url": server.url,
                        "headers": headers_json(&server.headers),
                    }),
                );
            }
            McpServer::Sse(server) => {
                mcp_servers.insert(
                    server.name.clone(),
                    json!({
                        "type": "sse",
                        "url": server.url,
                        "headers": headers_json(&server.headers),
                    }),
                );
            }
            _ => {}
        }
    }
    json!({ "mcpServers": mcp_servers })
}

fn headers_json(
    headers: &[agent_client_protocol::schema::HttpHeader],
) -> serde_json::Map<String, Value> {
    headers
        .iter()
        .map(|header| (header.name.clone(), Value::String(header.value.clone())))
        .collect()
}

impl Drop for ClaudePtySession {
    fn drop(&mut self) {
        drop(self.terminate());
        drop(self.reader_thread.take());
    }
}

fn reject_print_mode_args<'a>(args: impl IntoIterator<Item = &'a OsString>) -> anyhow::Result<()> {
    for arg in args {
        if arg == "-p" || arg == "--print" {
            return Err(anyhow!(
                "Claude PTY sessions must use interactive mode and cannot launch with {}",
                arg.to_string_lossy()
            ));
        }
        if arg.to_string_lossy().starts_with("--print=") {
            return Err(anyhow!(
                "Claude PTY sessions must use interactive mode and cannot launch with {}",
                arg.to_string_lossy()
            ));
        }
    }
    Ok(())
}
