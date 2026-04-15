use super::common::{ChatInput, FormattedText, ModelResponseCard, PhaseIndicator, PromptCard, PromptEditorModal, VoteDisplay, VoteTally, VoteTallyProps};
use crate::utils::{
    create_run_id, find_run_for_session, next_stream_event_with_cancel,
    recv_multi_event_with_cancel, register_active_run, remove_run, set_run_status,
    try_signal_read, try_signal_set, try_signal_update, ActiveRunRecord, ChatMessage,
    ChatHistory, ChatMode, ChatSession, CompetitiveHistory, InputSettings, Model,
    OpenRouterClient, RunStatus, SessionData, StreamEvent, Theme,
};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum CompetitivePromptType {
    Proposal,
    Voting,
}

impl CompetitivePromptType {
    fn name(&self) -> &'static str {
        match self {
            CompetitivePromptType::Proposal => "Proposal",
            CompetitivePromptType::Voting => "Voting",
        }
    }
    
    fn description(&self) -> &'static str {
        match self {
            CompetitivePromptType::Proposal => "Prompt sent to models to generate their proposals",
            CompetitivePromptType::Voting => "Prompt sent to models to vote for the best proposal",
        }
    }
    
    fn variables(&self) -> &'static str {
        match self {
            CompetitivePromptType::Proposal => "{user_question}",
            CompetitivePromptType::Voting => "{user_question}, {all_proposals}, {your_proposal}",
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
struct PromptTemplates {
    proposal: String,
    voting: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self {
            proposal: "You are participating in a competitive problem-solving challenge with other AI models. Provide your best solution to this question:\n\n{user_question}".to_string(),

            voting: "You are voting on the best solution among the proposals below. You CANNOT vote for your own response.\n\nUser Question: {user_question}\n\nAll Proposals:\n{all_proposals}\n\nYour Proposal:\n{your_proposal}\n\nVote for the BEST proposal by responding with ONLY the model ID of your choice (e.g., 'anthropic/claude-3.5-sonnet'). Do not vote for yourself.".to_string(),
        }
    }
}

impl PromptTemplates {
    fn get(&self, prompt_type: CompetitivePromptType) -> String {
        match prompt_type {
            CompetitivePromptType::Proposal => self.proposal.clone(),
            CompetitivePromptType::Voting => self.voting.clone(),
        }
    }
    
    fn set(&mut self, prompt_type: CompetitivePromptType, value: String) {
        match prompt_type {
            CompetitivePromptType::Proposal => self.proposal = value,
            CompetitivePromptType::Voting => self.voting = value,
        }
    }
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
struct CompetitiveRound {
    user_question: String,
    phase1_proposals: Vec<ModelProposal>,
    phase2_votes: Vec<ModelVote>,
    vote_tallies: Vec<VoteTally>,
    winners: Vec<String>,
    current_phase: CompetitivePhase,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CompetitivePhase {
    Proposal,
    Voting,
    Tallying,
    Complete,
}

impl CompetitivePhase {
    fn name(&self) -> &'static str {
        match self {
            CompetitivePhase::Proposal => "Phase 1: Proposals",
            CompetitivePhase::Voting => "Phase 2: Voting",
            CompetitivePhase::Tallying => "Tallying Results",
            CompetitivePhase::Complete => "Complete",
        }
    }

    fn badge_color(&self) -> &'static str {
        match self {
            CompetitivePhase::Proposal => "bg-blue-500",
            CompetitivePhase::Voting => "bg-purple-500",
            CompetitivePhase::Tallying => "bg-yellow-500",
            CompetitivePhase::Complete => "bg-gray-500",
        }
    }

    fn number(&self) -> u8 {
        match self {
            CompetitivePhase::Proposal => 1,
            CompetitivePhase::Voting => 2,
            CompetitivePhase::Tallying => 3,
            CompetitivePhase::Complete => 4,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_vote(response: &str, voter_id: &str, valid_model_ids: &[String]) -> Option<String> {
    let response = response.trim();

    // Try exact match first
    for model_id in valid_model_ids {
        if response.contains(model_id) {
            // Prevent self-voting
            if model_id == voter_id {
                return None;
            }
            return Some(model_id.clone());
        }
    }

    // Try fuzzy matching (last part of model ID)
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

    // Initialize all models with 0 votes
    for model_id in model_ids {
        tally_map.insert(model_id.clone(), (0, Vec::new()));
    }

    // Count votes
    for vote in votes {
        if let Some(voted_for) = &vote.voted_for {
            if let Some((count, voters)) = tally_map.get_mut(voted_for) {
                *count += 1;
                voters.push(vote.voter_id.clone());
            }
        }
    }

    // Convert to Vec<VoteTally>
    let mut tallies: Vec<VoteTally> = tally_map
        .into_iter()
        .map(|(model_id, (vote_count, voters))| VoteTally {
            model_id,
            vote_count,
            voters,
        })
        .collect();

    // Sort by vote count descending
    tallies.sort_by(|a, b| b.vote_count.cmp(&a.vote_count));

    // Find winners (all with max votes)
    let max_votes = tallies.first().map(|t| t.vote_count).unwrap_or(0);
    let winners: Vec<String> = tallies
        .iter()
        .filter(|t| t.vote_count == max_votes && max_votes > 0)
        .map(|t| t.model_id.clone())
        .collect();

    (tallies, winners)
}

// ============================================================================
// Main Component
// ============================================================================

#[component]
pub fn Competitive(theme: Signal<Theme>, client: Option<Arc<OpenRouterClient>>, input_settings: Signal<InputSettings>, session_id: Option<String>, on_session_saved: EventHandler<ChatSession>) -> Element {
    let active_runs = use_context::<Signal<HashMap<String, ActiveRunRecord>>>();
    // State
    let mut selected_models = use_signal(|| Vec::<String>::new());
    let mut selection_step = use_signal(|| 0usize); // 0 = select models, 1 = chat
    let mut conversation_history = use_signal(|| Vec::<CompetitiveRound>::new());
    let mut is_processing = use_signal(|| false);
    let mut current_streaming_responses = use_signal(|| HashMap::<String, String>::new());
    let mut current_phase = use_signal(|| CompetitivePhase::Proposal);
    let mut prompt_templates = use_signal(PromptTemplates::default);
    let mut current_run_id = use_signal(|| None::<String>);
    
    // Prompt editor state
    let mut prompt_editor_open = use_signal(|| false);
    let mut editing_prompt_type = use_signal(|| CompetitivePromptType::Proposal);
    let mut temp_prompt = use_signal(String::new);
    
    // Track the currently loaded session to avoid reloading on every render
    let mut loaded_session_id = use_signal(|| None::<String>);
    
    // Load history if session_id changes (not on every render)
    let session_id_for_load = session_id.clone();
    let session_loader = use_resource(move || {
        let session_id = session_id_for_load.clone();
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
        let current_sid = session_id.clone();
        if current_sid != *loaded_session_id.read() {
            match result {
                Ok(session_data) => {
                    if let ChatHistory::Competitive(history) = &session_data.history {
                        loaded_session_id.set(current_sid.clone());
                        let selected_models_clone = history.selected_models.clone();
                        selected_models.set(selected_models_clone.clone());
                        prompt_templates.set(PromptTemplates {
                            proposal: history.prompt_templates.proposal.clone(),
                            voting: history.prompt_templates.voting.clone(),
                        });
                        let converted_rounds: Vec<CompetitiveRound> = history
                            .rounds
                            .iter()
                            .map(|r| CompetitiveRound {
                                user_question: r.user_question.clone(),
                                phase1_proposals: r.phase1_proposals.iter().map(|p| ModelProposal {
                                    model_id: p.model_id.clone(),
                                    content: p.content.clone(),
                                    error_message: p.error_message.clone(),
                                }).collect(),
                                phase2_votes: r.phase2_votes.iter().map(|v| ModelVote {
                                    voter_id: v.voter_id.clone(),
                                    voted_for: v.voted_for.clone(),
                                    raw_response: v.raw_response.clone(),
                                    error_message: v.error_message.clone(),
                                }).collect(),
                                vote_tallies: r.vote_tallies.iter().map(|t| VoteTally {
                                    model_id: t.model_id.clone(),
                                    vote_count: t.vote_count,
                                    voters: t.voters.clone(),
                                }).collect(),
                                winners: r.winners.clone(),
                                current_phase: match r.current_phase.as_str() {
                                    "proposal" => CompetitivePhase::Proposal,
                                    "voting" => CompetitivePhase::Voting,
                                    "tallying" => CompetitivePhase::Tallying,
                                    _ => CompetitivePhase::Complete,
                                },
                            })
                            .collect();
                        conversation_history.set(converted_rounds);
                        if !selected_models_clone.is_empty() {
                            selection_step.set(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load session: {}", e);
                    loaded_session_id.set(current_sid);
                    selected_models.set(Vec::new());
                    conversation_history.set(Vec::new());
                    prompt_templates.set(PromptTemplates::default());
                    selection_step.set(0);
                }
            }
        }
    } else if session_id.is_none() && loaded_session_id.read().is_some() {
        loaded_session_id.set(None);
        selected_models.set(Vec::new());
        conversation_history.set(Vec::new());
        prompt_templates.set(PromptTemplates::default());
        selection_step.set(0);
    }

    // Search state for model selection
    let mut search_query = use_signal(|| String::new());
    let mut available_models = use_signal(|| None::<Result<Vec<Model>, String>>);

    let client_for_send = client.clone();
    
    // Prompt editor handlers
    let mut open_prompt_editor = move |ptype: CompetitivePromptType| {
        editing_prompt_type.set(ptype);
        temp_prompt.set(prompt_templates.read().get(ptype));
        prompt_editor_open.set(true);
    };
    
    let save_prompt = move |new_prompt: String| {
        let mut templates = prompt_templates.write();
        templates.set(*editing_prompt_type.read(), new_prompt);
    };

    let active_run_for_session = find_run_for_session(active_runs, &session_id, ChatMode::Competitive);
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

    // Cancel active run when component unmounts
    {
        let mut active_runs = active_runs.clone();
        let current_run_id = current_run_id.clone();
        use_drop(move || {
            if let Some(run_id) = current_run_id.try_read().ok().and_then(|id| id.clone()) {
                if let Some(run) = active_runs.try_read().ok().and_then(|runs| runs.get(&run_id).cloned()) {
                    run.request_cancel();
                    run.task.cancel();
                }
                remove_run(active_runs, &run_id);
            }
        });
    }

    // Fetch models on mount
    use_hook(|| {
        if let Some(client_arc) = &client {
            let client_clone = client_arc.clone();
            let mut models_clone = available_models.clone();
            spawn(async move {
                match client_clone.fetch_models().await {
                    Ok(models) => models_clone.set(Some(Ok(models))),
                    Err(e) => models_clone.set(Some(Err(e.to_string()))),
                }
            });
        }
    });

    // Handle message sending
    let mut send_message = move |user_msg: String| {
        if user_msg.trim().is_empty() || *is_processing.read() || run_is_active {
            return;
        }

        let models = selected_models();
        if models.len() < 2 {
            return;
        }

        if let Some(client_arc) = &client_for_send {
            is_processing.set(true);
            current_streaming_responses.write().clear();

            let client = client_arc.clone();
            let mut is_processing_clone = is_processing.clone();
            let mut conversation_history_clone = conversation_history.clone();
            let mut current_streaming_clone = current_streaming_responses.clone();
            let mut current_phase_clone = current_phase.clone();
            let templates = prompt_templates();
            let session_id_for_save = session_id.clone();
            let on_session_saved = on_session_saved.clone();
            let selected_models_for_save = selected_models.read().clone();
            let prompt_templates_for_save = prompt_templates.read().clone();
            let run_id = create_run_id(ChatMode::Competitive, &session_id);
            current_run_id.set(Some(run_id.clone()));
            let cancel_flag = Arc::new(AtomicBool::new(false));

            let run_id_for_task = run_id.clone();
            let mut active_runs_for_task = active_runs.clone();
            let cancel_flag_for_task = cancel_flag.clone();
            let task = spawn_forever(async move {
            // Create new round
            let mut round = CompetitiveRound {
                user_question: user_msg.clone(),
                phase1_proposals: Vec::new(),
                phase2_votes: Vec::new(),
                vote_tallies: Vec::new(),
                winners: Vec::new(),
                current_phase: CompetitivePhase::Proposal,
            };

            // PHASE 1: Proposals (Parallel)
            try_signal_set(&mut current_phase_clone, CompetitivePhase::Proposal);

            let proposal_prompt = templates.proposal.replace("{user_question}", &user_msg);
            let messages = vec![
                ChatMessage::system("You are in a competitive evaluation workflow. Follow the phase instructions exactly.".to_string()),
                ChatMessage::user(proposal_prompt),
            ];

            let mut phase1_results: HashMap<String, ModelProposal> = HashMap::new();

            match client.stream_chat_completion_multi(models.clone(), messages).await {
                Ok(mut rx) => {
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
                                    // Flush any remaining buffered content before marking done
                                    if let Some(accumulated) = content_buffer.remove(&model_id) {
                                        let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                            responses.insert(model_id.clone(), accumulated.clone());
                                        });
                                    }

                                let final_content = try_signal_read(&current_streaming_clone, |responses| {
                                    responses.get(&model_id).cloned().unwrap_or_default()
                                })
                                .unwrap_or_default();
                                phase1_results.insert(model_id.clone(), ModelProposal {
                                    model_id: model_id.clone(),
                                    content: final_content,
                                    error_message: None,
                                });
                                let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                    responses.remove(&model_id);
                                });
                            }
                            StreamEvent::Error(error) => {
                                if error == "Cancelled" {
                                    break;
                                }
                                phase1_results.insert(model_id.clone(), ModelProposal {
                                    model_id: model_id.clone(),
                                    content: String::new(),
                                    error_message: Some(error),
                                });
                                let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                    responses.remove(&model_id);
                                });
                            }
                        }
                    }

                    // Early exit if cancelled after Phase 1
                    if cancel_flag_for_task.load(Ordering::SeqCst) {
                        try_signal_set(&mut is_processing_clone, false);
                        set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("Error in Phase 1: {}", e);
                    try_signal_set(&mut is_processing_clone, false);
                    return;
                }
            }

            // Store proposals
            for model_id in &models {
                if let Some(proposal) = phase1_results.get(model_id) {
                    round.phase1_proposals.push(proposal.clone());
                }
            }

            // Filter successful proposals
            let successful_proposals: Vec<&ModelProposal> = round.phase1_proposals
                .iter()
                .filter(|p| p.error_message.is_none())
                .collect();

            if successful_proposals.len() < 2 {
                eprintln!("Not enough successful proposals");
                round.current_phase = CompetitivePhase::Complete;
                let _ = try_signal_update(&mut conversation_history_clone, |history| history.push(round));
                try_signal_set(&mut is_processing_clone, false);
                return;
            }

            // PHASE 2: Voting (Sequential)
            try_signal_set(&mut current_phase_clone, CompetitivePhase::Voting);
            round.current_phase = CompetitivePhase::Voting;

            // Build all_proposals text
            let all_proposals_text: String = successful_proposals
                .iter()
                .map(|p| format!("Model: {}\n{}\n", p.model_id, p.content))
                .collect::<Vec<_>>()
                .join("\n---\n\n");

            for model_id in &models {
                // Skip if this model didn't submit a successful proposal
                let my_proposal = match successful_proposals.iter().find(|p| &p.model_id == model_id) {
                    Some(p) => p,
                    None => continue,
                };

                let voting_prompt = templates.voting
                    .replace("{user_question}", &user_msg)
                    .replace("{all_proposals}", &all_proposals_text)
                    .replace("{your_proposal}", &my_proposal.content);

                let messages = vec![
                    ChatMessage::system("You are in a competitive evaluation workflow. Follow the phase instructions exactly.".to_string()),
                    ChatMessage::user(voting_prompt),
                ];
                let _ = try_signal_update(&mut current_streaming_clone, |responses| responses.clear());

                match client.stream_chat_completion(model_id.clone(), messages).await {
                    Ok(mut stream) => {
                        let mut vote_response = String::new();

                        // Throttle updates: only write to signal every 16ms
                        let mut last_update = std::time::Instant::now();
                        const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

                        while let Some(event) = next_stream_event_with_cancel(&mut stream, &cancel_flag_for_task).await {
                            if cancel_flag_for_task.load(Ordering::SeqCst) {
                                break;
                            }
                            match event {
                                StreamEvent::Content(content) => {
                                    vote_response.push_str(&content);

                                    // Throttle updates: only write to signal every 16ms
                                    if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                        let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                            responses.insert(model_id.clone(), vote_response.clone());
                                        });

                                        last_update = std::time::Instant::now();
                                    }
                                }
                                StreamEvent::Done => {
                                    // Flush final content and remove from streaming
                                    let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                        responses.insert(model_id.clone(), vote_response.clone());
                                        responses.remove(model_id.as_str());
                                    });

                                    let valid_model_ids: Vec<String> = successful_proposals.iter()
                                        .map(|p| p.model_id.clone())
                                        .collect();

                                    let voted_for = parse_vote(&vote_response, model_id, &valid_model_ids);

                                    round.phase2_votes.push(ModelVote {
                                        voter_id: model_id.clone(),
                                        voted_for,
                                        raw_response: vote_response.clone(),
                                        error_message: None,
                                    });
                                }
                                StreamEvent::Error(error) => {
                                    if error == "Cancelled" {
                                        break;
                                    }
                                    let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                        responses.remove(model_id);
                                    });

                                    round.phase2_votes.push(ModelVote {
                                        voter_id: model_id.clone(),
                                        voted_for: None,
                                        raw_response: String::new(),
                                        error_message: Some(error),
                                    });
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error voting for {}: {}", model_id, e);
                        round.phase2_votes.push(ModelVote {
                            voter_id: model_id.clone(),
                            voted_for: None,
                            raw_response: String::new(),
                            error_message: Some(e.to_string()),
                        });
                    }
                }

                // Early exit if cancelled during Phase 2
                if cancel_flag_for_task.load(Ordering::SeqCst) {
                    break;
                }
            }

            // Early exit if cancelled after Phase 2
            if cancel_flag_for_task.load(Ordering::SeqCst) {
                try_signal_set(&mut is_processing_clone, false);
                set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
                return;
            }

            // PHASE 3: Tallying
            try_signal_set(&mut current_phase_clone, CompetitivePhase::Tallying);
            round.current_phase = CompetitivePhase::Tallying;

            let valid_model_ids: Vec<String> = successful_proposals.iter()
                .map(|p| p.model_id.clone())
                .collect();

            let (tallies, winners) = compute_tallies(&round.phase2_votes, &valid_model_ids);
            round.vote_tallies = tallies;
            round.winners = winners;

            // Complete
            round.current_phase = CompetitivePhase::Complete;
            try_signal_set(&mut current_phase_clone, CompetitivePhase::Complete);
            let _ = try_signal_update(&mut conversation_history_clone, |history| history.push(round));
            try_signal_set(&mut is_processing_clone, false);
            
            // Auto-save only when there is content (spawn_blocking to avoid blocking async runtime)
            if let Some(sid) = session_id_for_save {
                let history = CompetitiveHistory {
                    rounds: try_signal_read(&conversation_history_clone, |history| history.clone())
                        .unwrap_or_default()
                        .iter()
                        .map(|r| crate::utils::CompetitiveRound {
                            user_question: r.user_question.clone(),
                            phase1_proposals: r.phase1_proposals.iter()
                                .map(|p| crate::utils::ModelProposal {
                                    model_id: p.model_id.clone(),
                                    content: p.content.clone(),
                                    error_message: p.error_message.clone(),
                                })
                                .collect(),
                            phase2_votes: r.phase2_votes.iter()
                                .map(|v| crate::utils::ModelVote {
                                    voter_id: v.voter_id.clone(),
                                    voted_for: v.voted_for.clone(),
                                    raw_response: v.raw_response.clone(),
                                    error_message: v.error_message.clone(),
                                })
                                .collect(),
                            vote_tallies: r.vote_tallies.iter()
                                .map(|t| crate::utils::VoteTally {
                                    model_id: t.model_id.clone(),
                                    vote_count: t.vote_count,
                                    voters: t.voters.clone(),
                                })
                                .collect(),
                            winners: r.winners.clone(),
                            current_phase: match r.current_phase {
                                CompetitivePhase::Proposal => "proposal",
                                CompetitivePhase::Voting => "voting",
                                CompetitivePhase::Tallying => "tallying",
                                CompetitivePhase::Complete => "complete",
                            }.to_string(),
                        })
                        .collect(),
                    selected_models: selected_models_for_save.clone(),
                    prompt_templates: crate::utils::PromptTemplates {
                        proposal: prompt_templates_for_save.proposal.clone(),
                        voting: prompt_templates_for_save.voting.clone(),
                    },
                };
                let history_enum = ChatHistory::Competitive(history.clone());
                if ChatHistory::has_content(&history_enum) {
                    let summary = ChatHistory::generate_chat_summary(&history_enum);
                    let session = ChatSession {
                        id: sid.clone(),
                        title: summary,
                        mode: ChatMode::Competitive,
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
            if cancel_flag_for_task.load(Ordering::SeqCst) {
                set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
            } else {
                remove_run(active_runs_for_task, &run_id_for_task);
            }
            });

            register_active_run(
                active_runs,
                run_id,
                session_id.clone(),
                ChatMode::Competitive,
                "Competitive round".to_string(),
                task,
                cancel_flag,
            );
        }
    };

    // Get filtered models for display
    let models_result = available_models.read();
    let loading = models_result.is_none();
    let (models_list, error) = match &*models_result {
        Some(Ok(models)) => (models.clone(), None),
        Some(Err(e)) => (Vec::new(), Some(e.clone())),
        None => (Vec::new(), None),
    };

    let filtered_models: Vec<Model> = {
        let search = search_query.read().to_lowercase();
        if search.is_empty() {
            models_list.clone()
        } else {
            models_list
                .iter()
                .filter(|m| {
                    m.display_name().to_lowercase().contains(&search)
                        || m.id.to_lowercase().contains(&search)
                })
                .cloned()
                .collect()
        }
    };

    // Render
    rsx! {
        div {
            class: "flex flex-col h-full",

            // Model selection or chat
            if *selection_step.read() == 0 {
                if let Some(_client_arc) = &client {
                    // Header
                    div {
                        class: "p-4 border-b border-[var(--color-base-300)]",
                        h2 {
                            class: "text-lg font-bold text-[var(--color-base-content)] mb-1",
                            "Select Models for Competition"
                        }
                        p {
                            class: "text-xs text-[var(--color-base-content)]/70",
                            "Choose 2 or more models to compete. They will propose solutions and vote for the best one."
                        }
                    }

                    // Search box
                    if !loading && error.is_none() {
                        div {
                            class: "px-4 pt-2",
                            input {
                                r#type: "text",
                                value: "{search_query}",
                                oninput: move |evt| search_query.set(evt.value().clone()),
                                placeholder: "Search models...",
                                class: "w-full px-3 py-2 text-sm rounded bg-[var(--color-base-100)] text-[var(--color-base-content)] border border-[var(--color-base-300)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent",
                            }
                        }
                    }

                    // Model list
                    div {
                        class: "flex-1 min-h-0 overflow-y-auto p-4",

                        if loading {
                            div {
                                class: "flex items-center justify-center h-full",
                                div {
                                    class: "text-center",
                                    div { class: "text-4xl mb-4", "⏳" }
                                    p {
                                        class: "text-[var(--color-base-content)]/70",
                                        "Loading available models..."
                                    }
                                }
                            }
                        } else if let Some(err) = &error {
                            div {
                                class: "flex items-center justify-center h-full",
                                div {
                                    class: "text-center max-w-md",
                                    div { class: "text-4xl mb-4", "⚠️" }
                                    p {
                                        class: "text-red-500 mb-2",
                                        "{err}"
                                    }
                                    p {
                                        class: "text-sm text-[var(--color-base-content)]/70",
                                        "Please check your API key in settings."
                                    }
                                }
                            }
                        } else {
                            div {
                                class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2",

                                for model in filtered_models.iter() {
                                    {
                                        let model_id = model.id.clone();
                                        let display_name = model.display_name();
                                        let is_selected = selected_models.read().contains(&model_id);

                                        rsx! {
                                            button {
                                                key: "{model_id}",
                                                onclick: move |_| {
                                                    let model_id = model_id.clone();
                                                    selected_models.write().retain(|id| id != &model_id);
                                                    if !is_selected {
                                                        selected_models.write().push(model_id);
                                                    }
                                                },
                                                class: if is_selected {
                                                    "p-3 rounded border-2 border-[var(--color-primary)] bg-[var(--color-primary)]/10 transition-all text-left"
                                                } else {
                                                    "p-3 rounded border border-[var(--color-base-300)] bg-[var(--color-base-200)] hover:border-[var(--color-primary)]/50 transition-all text-left"
                                                },
                                                div {
                                                    class: "flex items-start gap-2",
                                                    div {
                                                        class: "flex-shrink-0 mt-0.5",
                                                        if is_selected {
                                                            span { class: "text-[var(--color-primary)] text-sm", "✓" }
                                                        } else {
                                                            span { class: "text-[var(--color-base-content)]/30 text-sm", "○" }
                                                        }
                                                    }
                                                    div {
                                                        class: "flex-1 min-w-0",
                                                        div {
                                                            class: "font-semibold text-sm text-[var(--color-base-content)] truncate",
                                                            "{display_name}"
                                                        }
                                                        div {
                                                            class: "text-xs text-[var(--color-base-content)]/50 truncate",
                                                            "{model.id}"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if filtered_models.is_empty() {
                                div {
                                    class: "text-center py-8",
                                    p {
                                        class: "text-[var(--color-base-content)]/70",
                                        "No models found."
                                    }
                                }
                            }
                        }
                    }

                    // Footer buttons
                    if !loading && error.is_none() {
                        div {
                            class: "p-4 border-t border-[var(--color-base-300)]",
                            div {
                                class: "flex items-center justify-between gap-2",
                                div {
                                    class: "text-sm text-[var(--color-base-content)]/70",
                                    "{selected_models.read().len()} models selected (minimum 2)"
                                }
                                button {
                                    onclick: move |_| {
                                        if selected_models.read().len() >= 2 {
                                            selection_step.set(1);
                                        }
                                    },
                                    disabled: selected_models.read().len() < 2,
                                    class: "px-4 py-2 text-sm rounded bg-[var(--color-primary)] text-[var(--color-primary-content)] hover:bg-[var(--color-primary)]/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all",
                                    "Start Competition"
                                }
                            }
                        }
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
                // Prompt Templates Section
                div {
                    class: "p-3 border-b border-[var(--color-base-300)] bg-[var(--color-base-100)]",
                    
                    div {
                        class: "flex items-center justify-between mb-2",
                        h3 {
                            class: "text-sm font-semibold text-[var(--color-base-content)]",
                            "Prompt Templates (Click to customize)"
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
                        class: "grid grid-cols-1 md:grid-cols-2 gap-2",
                        
                        PromptCard {
                            theme,
                            title: "Proposal".to_string(),
                            phase_number: 1,
                            prompt: prompt_templates.read().proposal.clone(),
                            on_edit: move |_| open_prompt_editor(CompetitivePromptType::Proposal),
                        }
                        
                        PromptCard {
                            theme,
                            title: "Voting".to_string(),
                            phase_number: 2,
                            prompt: prompt_templates.read().voting.clone(),
                            on_edit: move |_| open_prompt_editor(CompetitivePromptType::Voting),
                        }
                    }
                }
                
                // Chat interface
                div {
                    class: "flex-1 min-h-0 overflow-y-auto p-4",

                    if conversation_history.read().is_empty() && !*is_processing.read() {
                        // Empty state
                        div {
                            class: "flex flex-col items-center justify-center h-full",
                            h2 {
                                class: "text-xl font-bold text-[var(--color-base-content)] mb-2",
                                "Competitive Mode Ready"
                            }
                            p {
                                class: "text-sm text-[var(--color-base-content)]/70 mb-4",
                                "Models will compete by proposing solutions and voting for the best one."
                            }
                            div {
                                class: "text-xs text-[var(--color-base-content)]/60 space-y-1 mb-4",
                                for (idx, model_id) in selected_models.read().iter().enumerate() {
                                    p { key: "{idx}", "• {model_id}" }
                                }
                            }
                            button {
                                onclick: move |_| {
                                    selection_step.set(0);
                                    conversation_history.write().clear();
                                },
                                class: "text-sm text-[var(--color-primary)] hover:underline",
                                "Change Models"
                            }
                        }
                    } else {
                        // Conversation display
                        div {
                            class: "space-y-6 w-full",

                        // Render each round
                        for round in conversation_history.read().iter() {
                            div {
                                class: "space-y-6",

                                // User question
                                div {
                                    class: "flex justify-end",
                                    div {
                                        class: "max-w-2xl p-3 sm:p-4 md:p-5 rounded-lg bg-[var(--color-primary)] text-[var(--color-primary-content)] text-sm sm:text-base",
                                        FormattedText {
                                            theme,
                                            content: round.user_question.clone(),
                                        }
                                    }
                                }

                                // Phase 1: Proposals
                                div {
                                    PhaseIndicator {
                                        phase_name: CompetitivePhase::Proposal.name().to_string(),
                                        phase_number: CompetitivePhase::Proposal.number(),
                                        badge_color: CompetitivePhase::Proposal.badge_color().to_string(),
                                    }

                                    div {
                                        class: "grid grid-cols-1 md:grid-cols-2 gap-4",

                                        for proposal in round.phase1_proposals.iter() {
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

                                // Phase 2: Voting
                                if !round.phase2_votes.is_empty() {
                                    div {
                                        PhaseIndicator {
                                            phase_name: CompetitivePhase::Voting.name().to_string(),
                                            phase_number: CompetitivePhase::Voting.number(),
                                            badge_color: CompetitivePhase::Voting.badge_color().to_string(),
                                        }

                                        div {
                                            class: "space-y-2",

                                            for vote in round.phase2_votes.iter() {
                                                VoteDisplay {
                                                    voter_id: vote.voter_id.clone(),
                                                    voted_for: vote.voted_for.clone(),
                                                    raw_response: Some(vote.raw_response.clone()),
                                                }
                                            }
                                        }
                                    }
                                }

                                // Phase 3: Results
                                if !round.vote_tallies.is_empty() {
                                    div {
                                        PhaseIndicator {
                                            phase_name: "Results".to_string(),
                                            phase_number: 3,
                                            badge_color: "bg-green-500".to_string(),
                                        }

                                        VoteTally {
                                            tallies: round.vote_tallies.iter()
                                                .map(|t| (t.model_id.clone(), t.vote_count, t.voters.clone()))
                                                .collect::<Vec<_>>(),
                                            winners: round.winners.clone(),
                                        }
                                    }
                                }
                            }
                        }

                            // Streaming indicators
                            if *is_processing.read() {
                                div {
                                    class: "space-y-6",

                                    match *current_phase.read() {
                                        CompetitivePhase::Proposal => rsx! {
                                            div {
                                                PhaseIndicator {
                                                    phase_name: CompetitivePhase::Proposal.name().to_string(),
                                                    phase_number: CompetitivePhase::Proposal.number(),
                                                    badge_color: CompetitivePhase::Proposal.badge_color().to_string(),
                                                }

                                                div {
                                                    class: "grid grid-cols-1 md:grid-cols-2 gap-4",

                                                    for model_id in selected_models.read().iter() {
                                                        {
                                                            let content = current_streaming_responses.read().get(model_id).cloned().unwrap_or_default();
                                                            let is_streaming = current_streaming_responses.read().contains_key(model_id);

                                                            rsx! {
                                                                ModelResponseCard {
                                                                    theme,
                                                                    model_id: model_id.clone(),
                                                                    content,
                                                                    error_message: None,
                                                                    is_streaming,
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        CompetitivePhase::Voting => rsx! {
                                            div {
                                                PhaseIndicator {
                                                    phase_name: CompetitivePhase::Voting.name().to_string(),
                                                    phase_number: CompetitivePhase::Voting.number(),
                                                    badge_color: CompetitivePhase::Voting.badge_color().to_string(),
                                                }

                                                div {
                                                    class: "text-sm text-[var(--color-base-content)]/70",
                                                    "Collecting votes..."
                                                }
                                            }
                                        },
                                        _ => rsx! {
                                            div {
                                                class: "text-sm text-[var(--color-base-content)]/70",
                                                "Finalizing results..."
                                            }
                                        },
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
                                "Cancelling active competitive run..."
                            } else {
                                "This competitive run is active in the background. You can cancel it manually."
                            }
                        }
                        button {
                            onclick: move |_| {
                                if let Some(run) = active_runs.read().get(&active_run_id).cloned() {
                                    run.request_cancel();
                                    set_run_status(active_runs, &active_run_id, RunStatus::Cancelling);
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
            
            // Prompt Editor Modal
            PromptEditorModal {
                theme,
                open: prompt_editor_open,
                prompt_type: match *editing_prompt_type.read() {
                    CompetitivePromptType::Proposal => super::common::PromptType::Initial,
                    CompetitivePromptType::Voting => super::common::PromptType::Review,
                },
                current_prompt: temp_prompt(),
                default_prompt: PromptTemplates::default().get(*editing_prompt_type.read()),
                on_save: save_prompt,
            }
        }
    }
}
