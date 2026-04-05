# ProjectPulse

Before ending every conversation, update a file called `projectpulse.md` in the project root with a brief, up-to-date summary of this workspace. This file is read by the ProjectPulse dashboard to give the user a quick overview across all their projects.

The file should contain:
- **Status**: One line summary of current project state (e.g., "Building REST API, auth module complete")
- **Last worked on**: What was done in this session
- **Next steps**: 1-3 bullet points of what needs to happen next
- **Blockers**: Any issues or dependencies blocking progress (or "None")
- **Key files**: The 3-5 most important files someone picking this up should look at

Keep it concise — this is a quick-glance summary, not documentation. Overwrite the previous content each time (don't append).


## Research Ideas

If the user asks to save a research idea, append it to the shared file:
`G:/My Drive/PythonCode/ProjectPulse/research_ideas.md`

Format each entry as:
```
## <Short title>
- **Date**: YYYY-MM-DD
- **Source workspace**: <this project's name>
- **Idea**: <1-3 sentence description>
- **Tags**: <comma-separated keywords>
```

Append below the marker line (`<!-- New ideas go below this line -->`), newest first. Do not overwrite existing entries.

## Conversation Memory (pulse_memory)

This workspace uses pulse_memory for persistent conversation history across Claude Code sessions.
Shared DB: all workspaces write to the same SQLite DB with daily auto-backups.

### At the START of each conversation
Check recent context for this workspace:
```
python "G:/My Drive/PythonCode/ProjectPulse/pulse_memory/recall.py" --recent --workspace TranslationProject --limit 5
```
Or search across ALL workspaces:
```
python "G:/My Drive/PythonCode/ProjectPulse/pulse_memory/recall.py" "search terms"
```

### At the END of each conversation
Save a summary with key messages:
```
python "G:/My Drive/PythonCode/ProjectPulse/pulse_memory/save_session.py" --workspace TranslationProject --summary "2-3 sentence summary" --messages "user: what was asked|assistant: what was done"
```
Tips for good messages:
- Include specific file names, function names, error messages, feature names
- These are what FTS5 search will match on in future sessions
- More detail = better recall later

