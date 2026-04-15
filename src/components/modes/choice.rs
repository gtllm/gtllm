use super::common::{ChatInput, FormattedText, Modal, ModelSelector, ModelResponseCard};
use crate::utils::{
    create_run_id, find_run_for_session, next_stream_event_with_cancel,
    recv_multi_event_with_cancel, register_active_run, remove_run, set_run_status,
    try_signal_read, try_signal_set, try_signal_update, ActiveRunRecord, ChatHistory,
    ChatMessage, ChatMode, ChatSession, InputSettings, OpenRouterClient, RunStatus, SessionData,
    StreamEvent, Theme,
};
use dioxus::prelude::*;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Debug, PartialEq)]
struct SystemPrompts {
    decision: String,
    collaborative: String,
    competitive: String,
}

impl Default for SystemPrompts {
    fn default() -> Self {
        Self {
            decision: "You are part of a team of AI models deciding on the best approach to answer a question. Consider whether collaboration or competition would yield better results.".to_string(),
            collaborative: "You are part of a collaborative AI team working together to provide the best answer.".to_string(),
            competitive: "You are in a competitive challenge. Provide your best solution and vote fairly for the best proposal.".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PromptEditTarget {
    Decision,
    Collaborative,
    Competitive,
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
enum Strategy {
    Collaborate,
    Compete,
}

#[derive(Clone, Debug, PartialEq)]
struct ModelDecision {
    model_id: String,
    decision: Option<Strategy>,
    reasoning: String,
    error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ModelResponse {
    model_id: String,
    content: String,
    error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ModelProposal {
    model_id: String,
    content: String,
    error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ModelVote {
    voter_id: String,
    voted_for: Option<String>,
    raw_response: String,
    error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct VoteTally {
    model_id: String,
    vote_count: usize,
    voters: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
enum ChoicePhase {
    Decision,           // LLMs deciding on strategy
    Collaborative,      // Executing collaborative workflow
    Competitive,       // Executing competitive workflow
    Complete,
}

impl ChoicePhase {
    fn name(&self) -> &'static str {
        match self {
            ChoicePhase::Decision => "Phase 1: Strategy Decision",
            ChoicePhase::Collaborative => "Phase 2: Collaborative Execution",
            ChoicePhase::Competitive => "Phase 2: Competitive Execution",
            ChoicePhase::Complete => "Complete",
        }
    }

    fn badge_color(&self) -> &'static str {
        match self {
            ChoicePhase::Decision => "bg-purple-500",
            ChoicePhase::Collaborative => "bg-blue-500",
            ChoicePhase::Competitive => "bg-orange-500",
            ChoicePhase::Complete => "bg-gray-500",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CollaborativeRound {
    user_question: String,
    phase1_responses: Vec<ModelResponse>,
    phase2_reviews: Vec<ModelResponse>,
    phase3_consensus: Option<ModelResponse>,
}

#[derive(Clone, Debug, PartialEq)]
struct CompetitiveRound {
    user_question: String,
    phase1_proposals: Vec<ModelProposal>,
    phase2_votes: Vec<ModelVote>,
    vote_tallies: Vec<VoteTally>,
    winners: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ChoiceRound {
    user_question: String,
    decisions: Vec<ModelDecision>,
    chosen_strategy: Option<Strategy>,
    collaborative_result: Option<CollaborativeRound>,
    competitive_result: Option<CompetitiveRound>,
    current_phase: ChoicePhase,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_decision(response: &str) -> Option<Strategy> {
    let response_lower = response.to_lowercase();
    
    // Look for explicit keywords
    if response_lower.contains("collaborat") && !response_lower.contains("compet") {
        return Some(Strategy::Collaborate);
    }
    if response_lower.contains("compet") && !response_lower.contains("collaborat") {
        return Some(Strategy::Compete);
    }
    
    // Look for vote-like patterns
    if response_lower.contains("vote") || response_lower.contains("choose") {
        if response_lower.contains("collaborat") {
            return Some(Strategy::Collaborate);
        }
        if response_lower.contains("compet") {
            return Some(Strategy::Compete);
        }
    }
    
    // Default: if unclear, try to infer from context
    // More "together", "joint", "unite" = collaborate
    // More "best", "win", "better" = compete
    let collaborate_keywords = ["together", "joint", "unite", "combine", "synthesize", "agree"];
    let compete_keywords = ["best", "win", "better", "superior", "outperform", "vote"];
    
    let collab_score = collaborate_keywords.iter()
        .filter(|kw| response_lower.contains(*kw))
        .count();
    let compete_score = compete_keywords.iter()
        .filter(|kw| response_lower.contains(*kw))
        .count();
    
    if collab_score > compete_score {
        Some(Strategy::Collaborate)
    } else if compete_score > collab_score {
        Some(Strategy::Compete)
    } else {
        None
    }
}

fn determine_strategy(decisions: &[ModelDecision]) -> Strategy {
    let mut collaborate_count = 0;
    let mut compete_count = 0;
    
    for decision in decisions {
        if let Some(Strategy::Collaborate) = decision.decision {
            collaborate_count += 1;
        } else if let Some(Strategy::Compete) = decision.decision {
            compete_count += 1;
        }
    }
    
    // Majority wins, default to collaborate if tie
    if compete_count > collaborate_count {
        Strategy::Compete
    } else {
        Strategy::Collaborate
    }
}

fn parse_vote(response: &str, voter_id: &str, valid_model_ids: &[String]) -> Option<String> {
    let response = response.trim();
    
    for model_id in valid_model_ids {
        if response.contains(model_id) {
            if model_id == voter_id {
                return None; // Can't vote for self
            }
            return Some(model_id.clone());
        }
    }
    
    // Try fuzzy matching
    for model_id in valid_model_ids {
        let model_name = model_id.split('/').last().unwrap_or(model_id);
        if response.to_lowercase().contains(&model_name.to_lowercase()) {
            if model_id == voter_id {
                return None;
            }
            return Some(model_id.clone());
        }
    }
    
    None
}

fn compute_tallies(votes: &[ModelVote], model_ids: &[String]) -> (Vec<VoteTally>, Vec<String>) {
    let mut tally_map: HashMap<String, (usize, Vec<String>)> = HashMap::new();
    
    for model_id in model_ids {
        tally_map.insert(model_id.clone(), (0, Vec::new()));
    }
    
    for vote in votes {
        if let Some(voted_for) = &vote.voted_for {
            if let Some((count, voters)) = tally_map.get_mut(voted_for) {
                *count += 1;
                voters.push(vote.voter_id.clone());
            }
        }
    }
    
    let mut tallies: Vec<VoteTally> = tally_map
        .into_iter()
        .map(|(model_id, (vote_count, voters))| VoteTally {
            model_id,
            vote_count,
            voters,
        })
        .collect();
    
    tallies.sort_by(|a, b| b.vote_count.cmp(&a.vote_count));
    
    let max_votes = tallies.first().map(|t| t.vote_count).unwrap_or(0);
    let winners: Vec<String> = tallies
        .iter()
        .filter(|t| t.vote_count == max_votes && max_votes > 0)
        .map(|t| t.model_id.clone())
        .collect();
    
    (tallies, winners)
}

// ============================================================================
// Component Props
// ============================================================================

#[derive(Props, Clone)]
pub struct ChoiceProps {
    theme: Signal<Theme>,
    client: Option<Arc<OpenRouterClient>>,
    input_settings: Signal<InputSettings>,
    session_id: Option<String>,
    on_session_saved: EventHandler<ChatSession>,
}

impl PartialEq for ChoiceProps {
    fn eq(&self, other: &Self) -> bool {
        self.theme == other.theme 
            && self.input_settings == other.input_settings
            && self.session_id == other.session_id
    }
}

// ============================================================================
// Main Component
// ============================================================================

#[component]
pub fn Choice(props: ChoiceProps) -> Element {
    let theme = props.theme;
    let client = props.client.clone();
    let client_for_send = props.client;
    let input_settings = props.input_settings;
    let active_runs = use_context::<Signal<HashMap<String, ActiveRunRecord>>>();
    let _ = theme.read();

    // Model selection state
    let mut selected_models = use_signal(|| Vec::<String>::new());
    let mut selection_step = use_signal(|| 0); // 0 = select models, 1 = chat

    // Chat state
    let mut conversation_history = use_signal(|| Vec::<ChoiceRound>::new());
    let mut is_processing = use_signal(|| false);
    let mut current_streaming_responses = use_signal(|| HashMap::<String, String>::new());
    let mut current_phase = use_signal(|| ChoicePhase::Decision);
    let mut current_run_id = use_signal(|| None::<String>);
    
    // System prompts
    let mut system_prompts = use_signal(SystemPrompts::default);
    let mut prompt_editor_open = use_signal(|| false);
    let mut editing_prompt_target = use_signal(|| PromptEditTarget::Decision);
    let mut temp_prompt = use_signal(String::new);
    
    // Track the currently loaded session to avoid reloading on every render
    let mut loaded_session_id = use_signal(|| None::<String>);
    
    // Load history if session_id changes (not on every render)
    let session_id = props.session_id.clone();
    let session_loader = use_resource(move || {
        let session_id = session_id.clone();
        async move {
            if let Some(sid) = session_id {
                let result = tokio::task::spawn_blocking(move || ChatHistory::load_session(&sid)).await;
                match result {
                    Ok(Ok(session_data)) => Some(Ok(session_data)),
                    Ok(Err(e)) => Some(Err(e)),
                    Err(e) => Some(Err(format!("Task join error: {}", e))),
                }
            } else {
                None
            }
        }
    });

    if let Some(Some(result)) = session_loader.read().as_ref() {
        let current_sid = props.session_id.clone();
        if current_sid != *loaded_session_id.read() {
            match result {
                Ok(session_data) => {
                    if let ChatHistory::LLMChoice(history) = &session_data.history {
                        loaded_session_id.set(current_sid.clone());
                        selected_models.set(history.selected_models.clone());
                        let converted_rounds: Vec<ChoiceRound> = history
                            .rounds
                            .iter()
                            .map(|r| {
                                let strategy = match r.decision.as_str() {
                                    "collaborate" => Some(Strategy::Collaborate),
                                    "compete" => Some(Strategy::Compete),
                                    _ => None,
                                };
                                ChoiceRound {
                                    user_question: r.user_message.clone(),
                                    decisions: vec![],
                                    chosen_strategy: strategy,
                                    collaborative_result: None,
                                    competitive_result: None,
                                    current_phase: ChoicePhase::Complete,
                                }
                            })
                            .collect();
                        conversation_history.set(converted_rounds);
                        if !history.selected_models.is_empty() {
                            selection_step.set(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load session: {}", e);
                    loaded_session_id.set(current_sid);
                    selected_models.set(Vec::new());
                    conversation_history.set(Vec::new());
                    system_prompts.set(SystemPrompts::default());
                    selection_step.set(0);
                }
            }
        }
    } else if props.session_id.is_none() && loaded_session_id.read().is_some() {
        loaded_session_id.set(None);
        selected_models.set(Vec::new());
        conversation_history.set(Vec::new());
        system_prompts.set(SystemPrompts::default());
        selection_step.set(0);
    }

    // Handle model selection
    let on_models_selected = move |models: Vec<String>| {
        selected_models.set(models);
        selection_step.set(1);
    };
    
    // Prompt editor handlers
    let mut open_prompt_editor = move |target: PromptEditTarget| {
        editing_prompt_target.set(target);
        let prompts = system_prompts.read();
        let current_prompt = match target {
            PromptEditTarget::Decision => prompts.decision.clone(),
            PromptEditTarget::Collaborative => prompts.collaborative.clone(),
            PromptEditTarget::Competitive => prompts.competitive.clone(),
        };
        temp_prompt.set(current_prompt);
        prompt_editor_open.set(true);
    };
    
    let save_prompt = move |_| {
        let mut prompts = system_prompts.write();
        match *editing_prompt_target.read() {
            PromptEditTarget::Decision => prompts.decision = temp_prompt(),
            PromptEditTarget::Collaborative => prompts.collaborative = temp_prompt(),
            PromptEditTarget::Competitive => prompts.competitive = temp_prompt(),
        }
        prompt_editor_open.set(false);
    };

    let active_run_for_session = find_run_for_session(active_runs, &props.session_id, ChatMode::LLMChoice);
    let run_is_active = active_run_for_session.as_ref().is_some_and(|run| {
        matches!(run.status, RunStatus::Running | RunStatus::Cancelling)
    });
    let cancel_bar_run = active_run_for_session.as_ref().and_then(|run| {
        if matches!(run.status, RunStatus::Running | RunStatus::Cancelling) {
            Some((run.id.clone(), matches!(run.status, RunStatus::Cancelling)))
        } else {
            None
        }
    });
    if let Some(active_run) = &active_run_for_session {
        if current_run_id.read().as_ref() != Some(&active_run.id) {
            current_run_id.set(Some(active_run.id.clone()));
        }
    } else if current_run_id.read().is_some() {
        current_run_id.set(None);
    }

    {
        let mut active_runs = active_runs.clone();
        let current_run_id = current_run_id.clone();
        use_drop(move || {
            if let Some(run_id) = current_run_id.try_read().ok().and_then(|id| id.clone()) {
                remove_run(active_runs, &run_id);
            }
        });
    }

    // Send message handler
    let send_message = move |text: String| {
        if text.trim().is_empty() || *is_processing.read() || run_is_active {
            return;
        }

        let models = selected_models.read().clone();
        if models.len() < 2 {
            return;
        }

        if let Some(client_arc) = &client_for_send {
            let client = client_arc.clone();
            let user_msg = text.clone();
            let prompts = system_prompts.read().clone();
            let mut is_processing_clone = is_processing.clone();
            let mut current_phase_clone = current_phase.clone();
            let mut current_streaming_clone = current_streaming_responses.clone();
            let mut conversation_history_clone = conversation_history.clone();
            let session_id_for_save = props.session_id.clone();
            let on_session_saved = props.on_session_saved.clone();
            let selected_models_for_save = selected_models.read().clone();
            let run_id = create_run_id(ChatMode::LLMChoice, &props.session_id);
            current_run_id.set(Some(run_id.clone()));
            let cancel_flag = Arc::new(AtomicBool::new(false));

            // Initialize new round
            {
                let mut history = conversation_history_clone.write();
                history.push(ChoiceRound {
                    user_question: user_msg.clone(),
                    decisions: vec![],
                    chosen_strategy: None,
                    collaborative_result: None,
                    competitive_result: None,
                    current_phase: ChoicePhase::Decision,
                });
            } // Drop the write borrow before spawning

            let run_id_for_task = run_id.clone();
            let mut active_runs_for_task = active_runs.clone();
            let cancel_flag_for_task = cancel_flag.clone();
            let task = spawn(async move {
                try_signal_set(&mut is_processing_clone, true);
                try_signal_set(&mut current_phase_clone, ChoicePhase::Decision);
                let _ = try_signal_update(&mut current_streaming_clone, |responses| responses.clear());

                // ========================================================
                // PHASE 1: Strategy Decision
                // ========================================================

                                let decision_prompt = format!(
                                    "User Question: {}\n\n\
                                    You have two options:\n\
                                    1. COLLABORATE: Work together to synthesize the best answer through discussion and consensus\n\
                                    2. COMPETE: Each model proposes a solution, then all models vote on the best one\n\n\
                                    Consider the nature of the question and decide which approach would yield better results.\n\n\
                                    Respond with your decision (COLLABORATE or COMPETE) and briefly explain your reasoning.",
                                    user_msg
                                );

                                let messages = vec![
                                    ChatMessage::system(prompts.decision.clone()),
                                    ChatMessage::user(decision_prompt)
                                ];
                let mut decisions: Vec<ModelDecision> = Vec::new();

                match client.stream_chat_completion_multi(models.clone(), messages).await {
                    Ok(mut rx) => {
                        let mut done_models = std::collections::HashSet::new();
                        let mut decision_responses: HashMap<String, String> = HashMap::new();

                        // Buffer content locally to throttle updates
                        let mut content_buffer: HashMap<String, String> = HashMap::new();
                        let mut last_update = std::time::Instant::now();
                        const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

                        while let Some(event) = recv_multi_event_with_cancel(&mut rx, &cancel_flag_for_task).await {
                            if cancel_flag_for_task.load(Ordering::SeqCst) {
                                break;
                            }
                            let model_id = event.model_id.clone();

                            match event.event {
                                StreamEvent::Content(content) => {
                                    // Accumulate in buffer instead of writing immediately
                                    content_buffer
                                        .entry(model_id.clone())
                                        .and_modify(|s| s.push_str(&content))
                                        .or_insert(content);

                                    // Throttle updates: only write to signal every 16ms
                                    if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                        // Flush only the active model to reduce cloning work.
                                        if let Some(accumulated) = content_buffer.get(&model_id) {
                                            let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                                responses.insert(model_id.clone(), accumulated.clone());
                                            });
                                        }

                                        last_update = std::time::Instant::now();
                                    }
                                }
                                StreamEvent::Done => {
                                    // Flush final accumulated content
                                    if let Some(accumulated) = content_buffer.get(&model_id) {
                                        let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                            responses.insert(model_id.clone(), accumulated.clone());
                                        });
                                    }

                                    let final_content = try_signal_read(&current_streaming_clone, |responses| {
                                        responses.get(&model_id).cloned().unwrap_or_default()
                                    })
                                    .unwrap_or_default();

                                    decision_responses.insert(model_id.clone(), final_content.clone());
                                    done_models.insert(model_id.clone());

                                    if done_models.len() >= models.len() {
                                        break;
                                    }
                                }
                                StreamEvent::Error(e) => {
                                    if e == "Cancelled" {
                                        break;
                                    }
                                    decisions.push(ModelDecision {
                                        model_id: model_id.clone(),
                                        decision: None,
                                        reasoning: String::new(),
                                        error_message: Some(e),
                                    });
                                    done_models.insert(model_id);
                                }
                            }
                        }

                        // Early exit if cancelled after decision phase
                        if cancel_flag_for_task.load(Ordering::SeqCst) {
                            try_signal_set(&mut is_processing_clone, false);
                            set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
                            return;
                        }

                        // Parse decisions
                        for model_id in &models {
                            if let Some(response) = decision_responses.get(model_id) {
                                let decision = parse_decision(response);
                                decisions.push(ModelDecision {
                                    model_id: model_id.clone(),
                                    decision: decision.clone(),
                                    reasoning: response.clone(),
                                    error_message: None,
                                });
                            }
                        }

                        // Determine strategy
                        let chosen_strategy = determine_strategy(&decisions);

                        // Update round with decisions
                        let _ = try_signal_update(&mut conversation_history_clone, |history| {
                            if let Some(last_round) = history.last_mut() {
                                last_round.decisions = decisions;
                                last_round.chosen_strategy = Some(chosen_strategy.clone());
                            }
                        });

                        let _ = try_signal_update(&mut current_streaming_clone, |responses| responses.clear());

                        // ========================================================
                        // PHASE 2: Execute Chosen Strategy
                        // ========================================================

                        match chosen_strategy {
                            Strategy::Collaborate => {
                                try_signal_set(&mut current_phase_clone, ChoicePhase::Collaborative);
                                let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                    if let Some(last_round) = history.last_mut() {
                                        last_round.current_phase = ChoicePhase::Collaborative;
                                    }
                                });

                                // Execute collaborative workflow
                                execute_collaborative(
                                    &client,
                                    &models,
                                    &user_msg,
                                    &prompts.collaborative,
                                    current_streaming_clone,
                                    conversation_history_clone,
                                    cancel_flag_for_task.clone(),
                                ).await;
                            }
                            Strategy::Compete => {
                                try_signal_set(&mut current_phase_clone, ChoicePhase::Competitive);
                                let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                    if let Some(last_round) = history.last_mut() {
                                        last_round.current_phase = ChoicePhase::Competitive;
                                    }
                                });

                                // Execute competitive workflow
                                execute_competitive(
                                    &client,
                                    &models,
                                    &user_msg,
                                    &prompts.competitive,
                                    current_streaming_clone,
                                    conversation_history_clone,
                                ).await;
                            }
                        }

                        try_signal_set(&mut current_phase_clone, ChoicePhase::Complete);
                        let _ = try_signal_update(&mut conversation_history_clone, |history| {
                            if let Some(last_round) = history.last_mut() {
                                last_round.current_phase = ChoicePhase::Complete;
                            }
                        });
                        try_signal_set(&mut is_processing_clone, false);
                        
                        // Auto-save only when there is content (spawn_blocking to avoid blocking async runtime)
                        if let Some(sid) = session_id_for_save {
                            let history_rounds: Vec<crate::utils::LLMChoiceRound> = try_signal_read(&conversation_history_clone, |history| history.clone())
                                .unwrap_or_default()
                                .iter()
                                .map(|r| {
                                    let decision = match r.chosen_strategy {
                                        Some(Strategy::Collaborate) => "collaborate".to_string(),
                                        Some(Strategy::Compete) => "compete".to_string(),
                                        None => "undecided".to_string(),
                                    };
                                    let content = r.collaborative_result.as_ref()
                                        .and_then(|cr| cr.phase3_consensus.as_ref().map(|c| c.content.clone()))
                                        .or_else(|| {
                                            r.competitive_result.as_ref()
                                                .and_then(|cr| cr.winners.first().map(|_| "Competitive round completed".to_string()))
                                        });
                                    crate::utils::LLMChoiceRound {
                                        user_message: r.user_question.clone(),
                                        decision,
                                        content,
                                    }
                                })
                                .collect();
                            let history = crate::utils::LLMChoiceHistory {
                                rounds: history_rounds,
                                selected_models: selected_models_for_save.clone(),
                            };
                            let history_enum = ChatHistory::LLMChoice(history.clone());
                            if ChatHistory::has_content(&history_enum) {
                                let summary = ChatHistory::generate_chat_summary(&history_enum);
                                let session = ChatSession {
                                    id: sid.clone(),
                                    title: summary,
                                    mode: ChatMode::LLMChoice,
                                    timestamp: ChatHistory::format_timestamp(),
                                };
                                let session_data = SessionData {
                                    session: session.clone(),
                                    history: history_enum,
                                    created_at: ChatHistory::session_timestamp_from_id(&sid)
                                        .unwrap_or_else(ChatHistory::format_timestamp),
                                    updated_at: ChatHistory::format_timestamp(),
                                };
                                match tokio::task::spawn_blocking(move || ChatHistory::save_session(&session_data)).await {
                                    Err(e) => eprintln!("Failed to save session task: {}", e),
                                    Ok(Err(e)) => eprintln!("Failed to save session: {}", e),
                                    Ok(Ok(_)) => on_session_saved.call(session),
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Handle error
                        let _ = try_signal_update(&mut conversation_history_clone, |history| {
                            if let Some(last_round) = history.last_mut() {
                                last_round.decisions = models
                                    .iter()
                                    .map(|id| ModelDecision {
                                        model_id: id.clone(),
                                        decision: None,
                                        reasoning: String::new(),
                                        error_message: Some(e.clone()),
                                    })
                                    .collect();
                            }
                        });
                        try_signal_set(&mut is_processing_clone, false);
                        
                        // Auto-save even on error (only when there is content; spawn_blocking to avoid blocking async runtime)
                        if let Some(sid) = session_id_for_save {
                            let history_rounds: Vec<crate::utils::LLMChoiceRound> = try_signal_read(&conversation_history_clone, |history| history.clone())
                                .unwrap_or_default()
                                .iter()
                                .map(|r| {
                                    let decision = match r.chosen_strategy {
                                        Some(Strategy::Collaborate) => "collaborate".to_string(),
                                        Some(Strategy::Compete) => "compete".to_string(),
                                        None => "undecided".to_string(),
                                    };
                                    let content = r.collaborative_result.as_ref()
                                        .and_then(|cr| cr.phase3_consensus.as_ref().map(|c| c.content.clone()))
                                        .or_else(|| {
                                            r.competitive_result.as_ref()
                                                .and_then(|cr| cr.winners.first().map(|_| "Competitive round completed".to_string()))
                                        });
                                    crate::utils::LLMChoiceRound {
                                        user_message: r.user_question.clone(),
                                        decision,
                                        content,
                                    }
                                })
                                .collect();
                            let history = crate::utils::LLMChoiceHistory {
                                rounds: history_rounds,
                                selected_models: selected_models_for_save.clone(),
                            };
                            let history_enum = ChatHistory::LLMChoice(history.clone());
                            if ChatHistory::has_content(&history_enum) {
                                let summary = ChatHistory::generate_chat_summary(&history_enum);
                                let session = ChatSession {
                                    id: sid.clone(),
                                    title: summary,
                                    mode: ChatMode::LLMChoice,
                                    timestamp: ChatHistory::format_timestamp(),
                                };
                                let session_data = SessionData {
                                    session: session.clone(),
                                    history: history_enum,
                                    created_at: ChatHistory::session_timestamp_from_id(&sid)
                                        .unwrap_or_else(ChatHistory::format_timestamp),
                                    updated_at: ChatHistory::format_timestamp(),
                                };
                                match tokio::task::spawn_blocking(move || ChatHistory::save_session(&session_data)).await {
                                    Err(e) => eprintln!("Failed to save session task: {}", e),
                                    Ok(Err(e)) => eprintln!("Failed to save session: {}", e),
                                    Ok(Ok(_)) => on_session_saved.call(session),
                                }
                            }
                        }
                    }
                }
                remove_run(active_runs_for_task, &run_id_for_task);
            });

            register_active_run(
                active_runs,
                run_id,
                props.session_id.clone(),
                ChatMode::LLMChoice,
                "LLM choice round".to_string(),
                task,
                cancel_flag,
            );
        }
    };

    rsx! {
        div {
            class: "flex flex-col h-full",

            // Model Selection Screen
            if *selection_step.read() == 0 {
                if let Some(client_arc) = &client {
                    ModelSelector {
                        theme,
                        client: client_arc.clone(),
                        on_models_selected,
                    }
                } else {
                    div {
                        class: "flex items-center justify-center h-full",
                        div {
                            class: "text-center",
                            p {
                                class: "text-[var(--color-base-content)]/70",
                                "No API client available"
                            }
                        }
                    }
                }
            } else {
                // System Prompts Section
                div {
                    class: "p-3 border-b border-[var(--color-base-300)] bg-[var(--color-base-100)]",
                    
                    div {
                        class: "flex items-center justify-between mb-2",
                        h3 {
                            class: "text-sm font-semibold text-[var(--color-base-content)]",
                            "System Prompts"
                        }
                        button {
                            onclick: move |_| {
                                selection_step.set(0);
                                conversation_history.write().clear();
                            },
                            class: "text-xs text-[var(--color-primary)] hover:underline",
                            "Change Models"
                        }
                    }
                    
                    div {
                        class: "grid grid-cols-1 md:grid-cols-3 gap-2",
                        
                        // Decision prompt
                        div {
                            class: "bg-[var(--color-base-200)] rounded p-2 border border-[var(--color-base-300)]",
                            div {
                                class: "flex items-center justify-between mb-1",
                                span {
                                    class: "text-xs font-semibold text-[var(--color-base-content)]",
                                    "Decision Phase"
                                }
                                button {
                                    onclick: move |_| open_prompt_editor(PromptEditTarget::Decision),
                                    class: "text-xs text-[var(--color-primary)] hover:underline",
                                    "Edit"
                                }
                            }
                            div {
                                class: "text-xs text-[var(--color-base-content)]/70 truncate",
                                "{system_prompts.read().decision}"
                            }
                        }
                        
                        // Collaborative prompt
                        div {
                            class: "bg-[var(--color-base-200)] rounded p-2 border border-[var(--color-base-300)]",
                            div {
                                class: "flex items-center justify-between mb-1",
                                span {
                                    class: "text-xs font-semibold text-[var(--color-base-content)]",
                                    "Collaborative"
                                }
                                button {
                                    onclick: move |_| open_prompt_editor(PromptEditTarget::Collaborative),
                                    class: "text-xs text-[var(--color-primary)] hover:underline",
                                    "Edit"
                                }
                            }
                            div {
                                class: "text-xs text-[var(--color-base-content)]/70 truncate",
                                "{system_prompts.read().collaborative}"
                            }
                        }
                        
                        // Competitive prompt
                        div {
                            class: "bg-[var(--color-base-200)] rounded p-2 border border-[var(--color-base-300)]",
                            div {
                                class: "flex items-center justify-between mb-1",
                                span {
                                    class: "text-xs font-semibold text-[var(--color-base-content)]",
                                    "Competitive"
                                }
                                button {
                                    onclick: move |_| open_prompt_editor(PromptEditTarget::Competitive),
                                    class: "text-xs text-[var(--color-primary)] hover:underline",
                                    "Edit"
                                }
                            }
                            div {
                                class: "text-xs text-[var(--color-base-content)]/70 truncate",
                                "{system_prompts.read().competitive}"
                            }
                        }
                    }
                }
                
                // Chat Interface
                div {
                    class: "flex-1 min-h-0 overflow-y-auto p-4",

                    if conversation_history.read().is_empty() {
                        // Empty state
                        div {
                            class: "flex flex-col items-center justify-center h-full",
                            h2 {
                                class: "text-xl font-bold text-[var(--color-base-content)] mb-2",
                                "🤖 LLM's Choice Mode Ready"
                            }
                            p {
                                class: "text-sm text-[var(--color-base-content)]/70 mb-4 text-center max-w-md",
                                "The selected LLMs will autonomously decide whether to collaborate or compete, then execute their chosen strategy."
                            }
                            div {
                                class: "text-xs text-[var(--color-base-content)]/60 space-y-1",
                                for (idx, model_id) in selected_models.read().iter().enumerate() {
                                    p { key: "{idx}", "• {model_id}" }
                                }
                            }
                            button {
                                onclick: move |_| {
                                    selection_step.set(0);
                                    conversation_history.write().clear();
                                },
                                class: "mt-4 text-sm text-[var(--color-primary)] hover:underline",
                                "Change Models"
                            }
                        }
                    } else {
                        // Conversation display
                        div {
                            class: "space-y-8 w-full",

                            for (round_idx, round) in conversation_history.read().iter().enumerate() {
                                div {
                                    key: "{round_idx}",

                                    // User message
                                    div {
                                        class: "flex justify-end mb-4",
                                        div {
                                            class: "max-w-[85%] bg-[var(--color-primary)] text-[var(--color-primary-content)] px-3 sm:px-4 md:px-5 py-2 sm:py-3 rounded-lg text-sm sm:text-base",
                                            FormattedText {
                                                theme,
                                                content: round.user_question.clone(),
                                            }
                                        }
                                    }

                                    // Decision Phase
                                    if !round.decisions.is_empty() {
                                        div {
                                            class: "mb-6",
                                            div {
                                                class: "flex items-center gap-2 mb-3",
                                                span {
                                                    class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {ChoicePhase::Decision.badge_color()}",
                                                    "{ChoicePhase::Decision.name()}"
                                                }
                                            }

                                            div {
                                                class: "grid grid-cols-1 md:grid-cols-2 gap-3 mb-4",
                                                for decision in round.decisions.iter() {
                                                    div {
                                                        key: "{decision.model_id}",
                                                        class: if decision.error_message.is_some() {
                                                            "bg-red-500/10 rounded-lg p-3 sm:p-4 border-2 border-red-500/50"
                                                        } else {
                                                            "bg-[var(--color-base-200)] rounded-lg p-3 sm:p-4 border border-[var(--color-base-300)]"
                                                        },
                                                        div {
                                                            class: "text-sm sm:text-base font-bold text-[var(--color-base-content)] mb-2",
                                                            "{decision.model_id}"
                                                        }
                                                        if let Some(error) = &decision.error_message {
                                                            div {
                                                                class: "text-sm sm:text-base text-red-500",
                                                                "Error: {error}"
                                                            }
                                                        } else {
                                                            div {
                                                                class: "mb-2",
                                                                if let Some(strategy) = &decision.decision {
                                                                    span {
                                                                        class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white",
                                                                        class: if *strategy == Strategy::Collaborate {
                                                                            "bg-blue-500"
                                                                        } else {
                                                                            "bg-orange-500"
                                                                        },
                                                                        if *strategy == Strategy::Collaborate {
                                                                            "✓ COLLABORATE"
                                                                        } else {
                                                                            "✓ COMPETE"
                                                                        }
                                                                    }
                                                                } else {
                                                                    span {
                                                                        class: "inline-block px-2 py-1 rounded text-xs font-semibold bg-gray-500 text-white",
                                                                        "UNDECIDED"
                                                                    }
                                                                }
                                                            }
                                                            div {
                                                                class: "text-xs text-[var(--color-base-content)]/70 whitespace-pre-wrap",
                                                                "{decision.reasoning}"
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Strategy announcement
                                            if let Some(strategy) = &round.chosen_strategy {
                                                {
                                                    let strategy_name = if *strategy == Strategy::Collaborate {
                                                        "COLLABORATE"
                                                    } else {
                                                        "COMPETE"
                                                    };
                                                    let strategy_desc = if *strategy == Strategy::Collaborate {
                                                        "The LLMs have decided to work together collaboratively."
                                                    } else {
                                                        "The LLMs have decided to compete and vote on the best solution."
                                                    };
                                                    rsx! {
                                                        div {
                                                            class: if *strategy == Strategy::Collaborate {
                                                                "bg-blue-500/10 rounded-lg p-4 border-2 border-blue-500/50"
                                                            } else {
                                                                "bg-orange-500/10 rounded-lg p-4 border-2 border-orange-500/50"
                                                            },
                                                            div {
                                                                class: "text-sm font-bold text-[var(--color-base-content)] mb-2",
                                                                "🎯 Chosen Strategy: {strategy_name}"
                                                            }
                                                            p {
                                                                class: "text-xs text-[var(--color-base-content)]/70",
                                                                "{strategy_desc}"
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Collaborative Results
                                    if let Some(collab) = &round.collaborative_result {
                                        div {
                                            class: "mb-6",
                                            div {
                                                class: "flex items-center gap-2 mb-3",
                                                span {
                                                    class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {ChoicePhase::Collaborative.badge_color()}",
                                                    "{ChoicePhase::Collaborative.name()}"
                                                }
                                            }

                                            // Phase 1: Initial Responses
                                            if !collab.phase1_responses.is_empty() {
                                                div {
                                                    class: "mb-4",
                                                    div {
                                                        class: "text-xs font-semibold text-[var(--color-base-content)]/70 mb-2",
                                                        "Initial Responses"
                                                    }
                                                    div {
                                                        class: "grid grid-cols-1 md:grid-cols-2 gap-3",
                                                        for response in collab.phase1_responses.iter() {
                                                            ModelResponseCard {
                                                                theme,
                                                                model_id: response.model_id.clone(),
                                                                content: response.content.clone(),
                                                                error_message: response.error_message.clone(),
                                                                is_streaming: false,
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Phase 2: Reviews
                                            if !collab.phase2_reviews.is_empty() {
                                                div {
                                                    class: "mb-4",
                                                    div {
                                                        class: "text-xs font-semibold text-[var(--color-base-content)]/70 mb-2",
                                                        "Cross-Reviews"
                                                    }
                                                    div {
                                                        class: "grid grid-cols-1 md:grid-cols-2 gap-3",
                                                        for review in collab.phase2_reviews.iter() {
                                                            ModelResponseCard {
                                                                theme,
                                                                model_id: review.model_id.clone(),
                                                                content: review.content.clone(),
                                                                error_message: review.error_message.clone(),
                                                                is_streaming: false,
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Phase 3: Consensus
                                            if let Some(consensus) = &collab.phase3_consensus {
                                                div {
                                                    class: "bg-green-500/10 rounded-lg p-4 border-2 border-green-500/50",
                                                    div {
                                                        class: "text-sm font-bold text-[var(--color-base-content)] mb-2",
                                                        "🎯 Collaborative Consensus (by {consensus.model_id})"
                                                    }
                                                    if let Some(error) = &consensus.error_message {
                                                        div {
                                                            class: "text-sm text-red-500",
                                                            "Error: {error}"
                                                        }
                                                    } else {
                                                        div {
                                                            class: "text-sm text-[var(--color-base-content)]",
                                                            FormattedText {
                                                                theme,
                                                                content: consensus.content.clone(),
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Competitive Results
                                    if let Some(comp) = &round.competitive_result {
                                        div {
                                            class: "mb-6",
                                            div {
                                                class: "flex items-center gap-2 mb-3",
                                                span {
                                                    class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {ChoicePhase::Competitive.badge_color()}",
                                                    "{ChoicePhase::Competitive.name()}"
                                                }
                                            }

                                            // Phase 1: Proposals
                                            if !comp.phase1_proposals.is_empty() {
                                                div {
                                                    class: "mb-4",
                                                    div {
                                                        class: "text-xs font-semibold text-[var(--color-base-content)]/70 mb-2",
                                                        "Proposals"
                                                    }
                                                    div {
                                                        class: "grid grid-cols-1 md:grid-cols-2 gap-3",
                                                        for proposal in comp.phase1_proposals.iter() {
                                                            ModelResponseCard {
                                                                theme,
                                                                model_id: proposal.model_id.clone(),
                                                                content: proposal.content.clone(),
                                                                error_message: proposal.error_message.clone(),
                                                                is_streaming: false,
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Phase 2: Votes
                                            if !comp.phase2_votes.is_empty() {
                                                div {
                                                    class: "mb-4",
                                                    div {
                                                        class: "text-xs font-semibold text-[var(--color-base-content)]/70 mb-2",
                                                        "Votes"
                                                    }
                                                    div {
                                                        class: "space-y-2",
                                                        for vote in comp.phase2_votes.iter() {
                                                            div {
                                                                key: "{vote.voter_id}",
                                                                class: "text-xs text-[var(--color-base-content)]/70",
                                                                "{vote.voter_id} voted for {vote.voted_for.as_ref().map(|v| v.as_str()).unwrap_or(\"none\")}"
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Results
                                            if !comp.vote_tallies.is_empty() {
                                                div {
                                                    class: "bg-yellow-500/10 rounded-lg p-4 border-2 border-yellow-500/50",
                                                    div {
                                                        class: "text-sm font-bold text-[var(--color-base-content)] mb-2",
                                                        "🏆 Voting Results"
                                                    }
                                                    div {
                                                        class: "space-y-2",
                                                        for tally in comp.vote_tallies.iter() {
                                                            div {
                                                                class: "text-xs text-[var(--color-base-content)]",
                                                                "{tally.model_id}: {tally.vote_count} vote(s)"
                                                            }
                                                        }
                                                    }
                                                    if !comp.winners.is_empty() {
                                                        div {
                                                            class: "mt-3 pt-3 border-t border-yellow-500/30",
                                                            div {
                                                                class: "text-sm font-bold text-[var(--color-base-content)]",
                                                                "Winner(s): {comp.winners.join(\", \")}"
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Streaming indicators
                            if *is_processing.read() && !current_streaming_responses.read().is_empty() {
                                div {
                                    class: "mb-6",
                                    div {
                                        class: "flex items-center gap-2 mb-3",
                                        span {
                                            class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {current_phase.read().badge_color()}",
                                            "{current_phase.read().name()}"
                                        }
                                        span {
                                            class: "inline-block w-2 h-2 bg-[var(--color-primary)] rounded-full animate-pulse"
                                        }
                                    }
                                    div {
                                        class: "grid grid-cols-1 md:grid-cols-2 gap-3",
                                        for (model_id, content) in current_streaming_responses.read().iter() {
                                            div {
                                                key: "{model_id}",
                                                class: "bg-[var(--color-base-200)] rounded-lg p-3 sm:p-4 border border-[var(--color-base-300)]",
                                                div {
                                                    class: "text-sm sm:text-base font-bold text-[var(--color-base-content)] mb-2 flex items-center gap-2",
                                                    span { "{model_id}" }
                                                    span {
                                                        class: "inline-block w-2 h-2 bg-[var(--color-primary)] rounded-full animate-pulse"
                                                    }
                                                }
                                                div {
                                                    class: "text-sm sm:text-base text-[var(--color-base-content)] whitespace-pre-wrap min-h-[3rem]",
                                                    "{content}"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some((active_run_id, is_cancelling)) = cancel_bar_run.clone() {
                    div {
                        class: "px-4 py-3 border-t border-[var(--color-base-300)] bg-[var(--color-base-100)] flex items-center justify-between gap-3",
                        div {
                            class: "text-sm text-[var(--color-base-content)]/70",
                            if is_cancelling {
                                "Cancelling active LLM choice run..."
                            } else {
                                "This LLM choice run is active in the background. You can cancel it manually."
                            }
                        }
                        button {
                            onclick: move |_| {
                                if let Some(run) = active_runs.read().get(&active_run_id).cloned() {
                                    run.request_cancel();
                                    run.task.cancel();
                                    set_run_status(active_runs, &active_run_id, RunStatus::Cancelling);
                                    remove_run(active_runs, &active_run_id);
                                    is_processing.set(false);
                                }
                            },
                            disabled: is_cancelling,
                            class: "px-3 py-2 rounded-lg bg-red-500 text-white text-sm font-medium hover:bg-red-500/90 disabled:opacity-60 disabled:cursor-not-allowed",
                            if is_cancelling { "Cancelling..." } else { "Cancel Run" }
                        }
                    }
                }

                // Input area
                ChatInput {
                    theme,
                    input_settings,
                    on_send: send_message,
                }
            }
            
            // System Prompt Editor Modal
            Modal {
                theme,
                open: prompt_editor_open,
                on_close: move |_| {
                    prompt_editor_open.set(false);
                },
                
                div {
                    class: "p-6",
                    
                    // Header
                    div {
                        class: "flex items-start justify-between mb-4",
                        div {
                            h2 {
                                class: "text-xl font-bold text-[var(--color-base-content)]",
                                {
                                    let prompt_name = match *editing_prompt_target.read() {
                                        PromptEditTarget::Decision => "Decision Phase",
                                        PromptEditTarget::Collaborative => "Collaborative",
                                        PromptEditTarget::Competitive => "Competitive",
                                    };
                                    format!("Edit {} System Prompt", prompt_name)
                                }
                            }
                            p {
                                class: "text-sm text-[var(--color-base-content)]/70 mt-1",
                                {
                                    match *editing_prompt_target.read() {
                                        PromptEditTarget::Decision => "Sets behavior when LLMs decide on their strategy",
                                        PromptEditTarget::Collaborative => "Sets behavior for collaborative execution",
                                        PromptEditTarget::Competitive => "Sets behavior for competitive execution",
                                    }
                                }
                            }
                        }
                        button {
                            class: "text-2xl text-[var(--color-base-content)]/70 hover:text-[var(--color-base-content)] transition-colors",
                            onclick: move |_| {
                                prompt_editor_open.set(false);
                            },
                            "×"
                        }
                    }
                    
                    // Prompt editor textarea
                    div {
                        class: "mb-4",
                        textarea {
                            value: "{temp_prompt}",
                            oninput: move |evt| temp_prompt.set(evt.value()),
                            rows: "10",
                            class: "w-full p-3 border-2 rounded-lg font-mono text-sm bg-[var(--color-base-100)] text-[var(--color-base-content)] border-[var(--color-base-300)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent resize-y min-h-[200px]",
                            placeholder: "Enter system prompt...",
                            autofocus: true,
                        }
                    }
                    
                    // Character count
                    div {
                        class: "text-xs text-[var(--color-base-content)]/50 mb-4 text-right",
                        "{temp_prompt.read().len()} characters"
                    }
                    
                    // Action buttons
                    div {
                        class: "flex justify-between items-center gap-3",
                        button {
                            onclick: move |_| {
                                let defaults = SystemPrompts::default();
                                let default_prompt = match *editing_prompt_target.read() {
                                    PromptEditTarget::Decision => defaults.decision,
                                    PromptEditTarget::Collaborative => defaults.collaborative,
                                    PromptEditTarget::Competitive => defaults.competitive,
                                };
                                temp_prompt.set(default_prompt);
                            },
                            class: "px-4 py-2 text-sm rounded border border-[var(--color-base-300)] bg-[var(--color-base-200)] text-[var(--color-base-content)] hover:bg-[var(--color-base-300)] transition-colors",
                            "Reset to Default"
                        }
                        div {
                            class: "flex gap-2",
                            button {
                                onclick: move |_| {
                                    prompt_editor_open.set(false);
                                },
                                class: "px-4 py-2 text-sm rounded border border-[var(--color-base-300)] bg-[var(--color-base-200)] text-[var(--color-base-content)] hover:bg-[var(--color-base-300)] transition-colors",
                                "Cancel"
                            }
                            button {
                                onclick: save_prompt,
                                class: "px-4 py-2 text-sm rounded bg-[var(--color-primary)] text-[var(--color-primary-content)] hover:bg-[var(--color-primary)]/90 transition-colors font-medium",
                                "Save Prompt"
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Workflow Execution Functions
// ============================================================================

async fn execute_collaborative(
    client: &Arc<OpenRouterClient>,
    models: &[String],
    user_msg: &str,
    system_prompt: &str,
    mut current_streaming: Signal<HashMap<String, String>>,
    mut conversation_history: Signal<Vec<ChoiceRound>>,
    cancel_flag: Arc<AtomicBool>,
) {
    // Phase 1: Initial Responses
    let initial_prompt = format!(
        "Provide your best answer to this question:\n\n{}",
        user_msg
    );
    let messages = vec![
        ChatMessage::system(system_prompt.to_string()),
        ChatMessage::user(initial_prompt)
    ];

    let mut phase1_results: HashMap<String, ModelResponse> = HashMap::new();

    if let Ok(mut rx) = client.stream_chat_completion_multi(models.to_vec(), messages).await {
        let mut done_models = std::collections::HashSet::new();
        
        // Buffer content locally to throttle updates
        let mut content_buffer: HashMap<String, String> = HashMap::new();
        let mut last_update = std::time::Instant::now();
        const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

        while let Some(event) = rx.recv().await {
            let model_id = event.model_id.clone();

            match event.event {
                StreamEvent::Content(content) => {
                    // Accumulate in buffer instead of writing immediately
                    content_buffer
                        .entry(model_id.clone())
                        .and_modify(|s| s.push_str(&content))
                        .or_insert(content);
                    
                    // Throttle updates: only write to signal every 16ms
                    if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                        // Flush only the active model to reduce cloning work.
                        if let Some(accumulated) = content_buffer.get(&model_id) {
                            let _ = try_signal_update(&mut current_streaming, |responses| {
                                responses.insert(model_id.clone(), accumulated.clone());
                            });
                        }
                        
                        last_update = std::time::Instant::now();
                    }
                }
                StreamEvent::Done => {
                    // Flush final accumulated content
                    if let Some(accumulated) = content_buffer.get(&model_id) {
                        let _ = try_signal_update(&mut current_streaming, |responses| {
                            responses.insert(model_id.clone(), accumulated.clone());
                        });
                    }
                    
                    let final_content = try_signal_read(&current_streaming, |responses| {
                        responses.get(&model_id).cloned().unwrap_or_default()
                    })
                    .unwrap_or_default();

                    phase1_results.insert(
                        model_id.clone(),
                        ModelResponse {
                            model_id: model_id.clone(),
                            content: final_content,
                            error_message: None,
                        },
                    );
                    done_models.insert(model_id);

                    if done_models.len() >= models.len() {
                        break;
                    }
                }
                StreamEvent::Error(e) => {
                    phase1_results.insert(
                        model_id.clone(),
                        ModelResponse {
                            model_id: model_id.clone(),
                            content: String::new(),
                            error_message: Some(e),
                        },
                    );
                    done_models.insert(model_id);
                }
            }
        }
    }

    let _ = try_signal_update(&mut current_streaming, |responses| responses.clear());

    // Prepare Phase 1 responses (drop borrow quickly)
    let phase1_responses: Vec<ModelResponse> = {
        models
            .iter()
            .filter_map(|id| phase1_results.get(id).cloned())
            .collect()
    };

    // Phase 2: Cross-Review (simplified - just collect reviews)
    let mut phase2_reviews = Vec::new();
    let successful_phase1: Vec<_> = phase1_results
        .values()
        .filter(|r| r.error_message.is_none())
        .collect();

    if successful_phase1.len() >= 2 {
        for model_id in models {
            let other_responses: String = successful_phase1
                .iter()
                .filter(|r| &r.model_id != model_id)
                .map(|r| format!("{}: {}", r.model_id, r.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            let review_prompt = format!(
                "Review the following responses from other AI models. Provide constructive feedback.\n\nUser Question: {}\n\nOther responses:\n{}\n\nProvide your analysis:",
                user_msg, other_responses
            );

            let review_messages = vec![
                ChatMessage::system(system_prompt.to_string()),
                ChatMessage::user(review_prompt)
            ];

            if let Ok(mut stream) = client.stream_chat_completion(model_id.clone(), review_messages).await {
                let mut review_content = String::new();
                while let Some(event) = stream.next().await {
                    match event {
                        StreamEvent::Content(content) => {
                            review_content.push_str(&content);
                        }
                        StreamEvent::Done => {
                            phase2_reviews.push(ModelResponse {
                                model_id: model_id.clone(),
                                content: review_content,
                                error_message: None,
                            });
                            break;
                        }
                        StreamEvent::Error(e) => {
                            phase2_reviews.push(ModelResponse {
                                model_id: model_id.clone(),
                                content: String::new(),
                                error_message: Some(e),
                            });
                            break;
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Consensus
    let synthesizer_id = &models[0];
    let initial_responses_text: String = successful_phase1
        .iter()
        .map(|r| format!("{}: {}", r.model_id, r.content))
        .collect::<Vec<_>>()
        .join("\n\n");

    let reviews_text: String = phase2_reviews
        .iter()
        .filter(|r| r.error_message.is_none())
        .map(|r| format!("{}: {}", r.model_id, r.content))
        .collect::<Vec<_>>()
        .join("\n\n");

    let consensus_prompt = format!(
        "Based on all the initial responses and reviews below, synthesize a final collaborative answer.\n\nUser Question: {}\n\nInitial Responses:\n{}\n\nReviews:\n{}\n\nSynthesize the best collaborative answer:",
        user_msg, initial_responses_text, reviews_text
    );

    let consensus_messages = vec![
        ChatMessage::system(system_prompt.to_string()),
        ChatMessage::user(consensus_prompt)
    ];
    let mut consensus_content = String::new();
    let mut consensus_error: Option<String> = None;

    match client.stream_chat_completion(synthesizer_id.clone(), consensus_messages).await {
        Ok(mut stream) => {
            while let Some(event) = stream.next().await {
                match event {
                    StreamEvent::Content(content) => {
                        consensus_content.push_str(&content);
                    }
                    StreamEvent::Done => {
                        break;
                    }
                    StreamEvent::Error(e) => {
                        consensus_error = Some(e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            consensus_error = Some(e);
        }
    }

    // Update round with all results (get fresh borrow)
    let _ = try_signal_update(&mut conversation_history, |history| {
        if let Some(last_round) = history.last_mut() {
            last_round.collaborative_result = Some(CollaborativeRound {
                user_question: user_msg.to_string(),
                phase1_responses,
                phase2_reviews,
                phase3_consensus: Some(ModelResponse {
                    model_id: synthesizer_id.clone(),
                    content: consensus_content,
                    error_message: consensus_error,
                }),
            });
        }
    });
}

async fn execute_competitive(
    client: &Arc<OpenRouterClient>,
    models: &[String],
    user_msg: &str,
    system_prompt: &str,
    mut current_streaming: Signal<HashMap<String, String>>,
    mut conversation_history: Signal<Vec<ChoiceRound>>,
) {
    // Phase 1: Proposals
    let proposal_prompt = format!(
        "Provide your best solution:\n\n{}",
        user_msg
    );
    let messages = vec![
        ChatMessage::system(system_prompt.to_string()),
        ChatMessage::user(proposal_prompt)
    ];

    let mut phase1_results: HashMap<String, ModelProposal> = HashMap::new();

    if let Ok(mut rx) = client.stream_chat_completion_multi(models.to_vec(), messages).await {
        // Buffer content locally to throttle updates
        let mut content_buffer: HashMap<String, String> = HashMap::new();
        let mut last_update = std::time::Instant::now();
        const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

        while let Some(event) = rx.recv().await {
            let model_id = event.model_id.clone();

            match event.event {
                StreamEvent::Content(content) => {
                    // Accumulate in buffer instead of writing immediately
                    content_buffer
                        .entry(model_id.clone())
                        .and_modify(|s| s.push_str(&content))
                        .or_insert(content);
                    
                    // Throttle updates: only write to signal every 16ms
                    if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                        // Flush only the active model to reduce cloning work.
                        if let Some(accumulated) = content_buffer.get(&model_id) {
                            let _ = try_signal_update(&mut current_streaming, |responses| {
                                responses.insert(model_id.clone(), accumulated.clone());
                            });
                        }
                        
                        last_update = std::time::Instant::now();
                    }
                }
                StreamEvent::Done => {
                    // Flush final accumulated content
                    if let Some(accumulated) = content_buffer.get(&model_id) {
                        let _ = try_signal_update(&mut current_streaming, |responses| {
                            responses.insert(model_id.clone(), accumulated.clone());
                        });
                    }
                    
                    let final_content = try_signal_read(&current_streaming, |responses| {
                        responses.get(&model_id).cloned().unwrap_or_default()
                    })
                    .unwrap_or_default();
                    phase1_results.insert(
                        model_id.clone(),
                        ModelProposal {
                            model_id: model_id.clone(),
                            content: final_content,
                            error_message: None,
                        },
                    );
                    let _ = try_signal_update(&mut current_streaming, |responses| {
                        responses.remove(&model_id);
                    });
                }
                StreamEvent::Error(error) => {
                    phase1_results.insert(
                        model_id.clone(),
                        ModelProposal {
                            model_id: model_id.clone(),
                            content: String::new(),
                            error_message: Some(error),
                        },
                    );
                    let _ = try_signal_update(&mut current_streaming, |responses| {
                        responses.remove(&model_id);
                    });
                }
            }
        }
    }

    let _ = try_signal_update(&mut current_streaming, |responses| responses.clear());

    // Phase 2: Voting
    let successful_proposals: Vec<&ModelProposal> = phase1_results
        .values()
        .filter(|p| p.error_message.is_none())
        .collect();

    if successful_proposals.len() >= 2 {
        let all_proposals_text: String = successful_proposals
            .iter()
            .map(|p| format!("{}: {}", p.model_id, p.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut phase2_votes = Vec::new();

        for model_id in models {
            let your_proposal = phase1_results
                .get(model_id)
                .map(|p| p.content.clone())
                .unwrap_or_default();

            let voting_prompt = format!(
                "You are voting on the best solution. You CANNOT vote for your own response.\n\nUser Question: {}\n\nAll Proposals:\n{}\n\nYour Proposal:\n{}\n\nVote for the BEST proposal by responding with ONLY the model ID of your choice.",
                user_msg, all_proposals_text, your_proposal
            );

            let voting_messages = vec![
                ChatMessage::system(system_prompt.to_string()),
                ChatMessage::user(voting_prompt)
            ];

            if let Ok(mut stream) = client.stream_chat_completion(model_id.clone(), voting_messages).await {
                let mut vote_response = String::new();
                while let Some(event) = stream.next().await {
                    match event {
                        StreamEvent::Content(content) => {
                            vote_response.push_str(&content);
                        }
                        StreamEvent::Done => {
                            let voted_for = parse_vote(&vote_response, model_id, models);
                            phase2_votes.push(ModelVote {
                                voter_id: model_id.clone(),
                                voted_for,
                                raw_response: vote_response,
                                error_message: None,
                            });
                            break;
                        }
                        StreamEvent::Error(e) => {
                            phase2_votes.push(ModelVote {
                                voter_id: model_id.clone(),
                                voted_for: None,
                                raw_response: String::new(),
                                error_message: Some(e),
                            });
                            break;
                        }
                    }
                }
            }
        }

        // Compute tallies
        let (vote_tallies, winners) = compute_tallies(&phase2_votes, models);

        // Update round
        let _ = try_signal_update(&mut conversation_history, |history| {
            if let Some(last_round) = history.last_mut() {
                last_round.competitive_result = Some(CompetitiveRound {
                    user_question: user_msg.to_string(),
                    phase1_proposals: models
                        .iter()
                        .filter_map(|id| phase1_results.get(id).cloned())
                        .collect(),
                    phase2_votes,
                    vote_tallies,
                    winners,
                });
            }
        });
    }
}
