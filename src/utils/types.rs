use dioxus::core::Task;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AppView {
    NewChat,
    ChatMode(ChatMode),
    Settings,
}

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatMode {
    #[serde(rename = "standard")]
    Standard,
    #[serde(rename = "pvp")]
    PvP,
    #[serde(rename = "collaborative")]
    Collaborative,
    #[serde(rename = "competitive")]
    Competitive,
    #[serde(rename = "llm_choice")]
    LLMChoice,
}

impl ChatMode {
    pub fn name(&self) -> &'static str {
        match self {
            ChatMode::Standard => "Standard",
            ChatMode::PvP => "PvP",
            ChatMode::Collaborative => "Collaborative",
            ChatMode::Competitive => "Competitive",
            ChatMode::LLMChoice => "LLM's Choice",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            ChatMode::Standard => "Single LLM chat",
            ChatMode::PvP => "2 bots compete, 1 moderator judges",
            ChatMode::Collaborative => "Multiple bots jointly agree on best solution",
            ChatMode::Competitive => "All bots vote for the best (can't vote for their own)",
            ChatMode::LLMChoice => "LLMs decide to collaborate or compete",
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Message {
    pub id: usize,
    pub content: String,
    pub sender: String,
    pub is_user: bool,
    pub timestamp: String,
}

#[derive(Clone, PartialEq, Debug)]
pub struct ArenaMessage {
    pub id: usize,
    pub content: String,
    pub bot_name: String,
    pub timestamp: String,
    pub vote_count: Option<usize>,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String, // UUID
    pub title: String,
    pub mode: ChatMode,
    pub timestamp: String,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InputSettings {
    pub ctrl_enter_submit: bool, // true = Ctrl+Enter to submit, false = Enter to submit
}

#[derive(Clone, PartialEq, Debug)]
pub enum RunStatus {
    Running,
    Cancelling,
    Completed,
    Cancelled,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct ActiveRunRecord {
    pub id: String,
    pub session_id: Option<String>,
    pub mode: ChatMode,
    pub label: String,
    pub status: RunStatus,
    pub started_at: String,
    pub task: Task,
    pub cancel_flag: Arc<AtomicBool>,
}

impl ActiveRunRecord {
    pub fn request_cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }
}

impl PartialEq for ActiveRunRecord {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.session_id == other.session_id
            && self.mode == other.mode
            && self.label == other.label
            && self.status == other.status
            && self.started_at == other.started_at
    }
}
