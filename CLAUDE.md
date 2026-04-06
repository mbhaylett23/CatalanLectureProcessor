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

pulse_memory provides persistent memory across Claude Code sessions.
Shared DB: all workspaces write to the same SQLite DB with daily auto-backups.

### At the START of each conversation (MANDATORY)
Run this FIRST before doing anything else:
```
python "G:/My Drive/PythonCode/ProjectPulse/pulse_memory/run.py" startup --workspace TranslationProject
```
This loads recent sessions, curated memory, available skills, wiki stats, and knowledge counts.

To search for specific past topics:
```
python "G:/My Drive/PythonCode/ProjectPulse/pulse_memory/run.py" recall "search terms"
```

### At the END of each conversation (MANDATORY)
Save a summary with key messages:
```
python "G:/My Drive/PythonCode/ProjectPulse/pulse_memory/run.py" save_session --workspace TranslationProject --summary "2-3 sentence summary" --messages "user: what was asked|assistant: what was done"
```
Include specific file names, function names, error messages in messages for FTS5 search.

### Wiki (knowledge pages)
The wiki is an LLM-maintained knowledge base at `G:/My Drive/PythonCode/ProjectPulse/pulse_memory/wiki/`. It contains topic and entity pages that compound over time.
- Search wiki: `from pulse_memory import Wiki; w = Wiki(); w.search("query")`
- Ingest finding: `w.ingest("finding text", tags=["tag1"], source="attribution")`
- After complex research sessions, ingest key findings into the wiki

### Skills (reusable procedures)
Skills are at `G:/My Drive/PythonCode/ProjectPulse/pulse_memory/skills/`. View available skills before starting complex tasks:
```python
from pulse_memory import SkillStore
skills = SkillStore()
skills.list_skills()           # see all skills
skills.view_skill("name")     # read full procedure
```
After completing a complex task (5+ steps), create a skill so future sessions can reuse it.


