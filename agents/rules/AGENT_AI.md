# Agent AI

- Behavior

  - Rule
  - Skill
  - Knowledge bases
- Agent AI should be your second brain, including memory, as in this stage, a dev + AI agent can result in a lot of stuff done, but lead to burnt out; therefore, must make enough document / .md files for the Agent AI to have the grasp of everything in the project.

-- Behavior layer: prompting via .md files

- .MD files is a description that contain:

  - name
  - background
  - goals/purpose
  - Input/Output
  - How to do
- From the description -> make general plan -> Pipeline -> Details experiment plan (all this should be in a single .md files) -> Progress status (in it individual .md files): to do, in progress, Done, On hold, Canceled
- A .MD that note out where everything are using Relative link (agents/OVERVIEW.md: for internal things in project), Cross-file anchor link (agents/OVERVIEW.md#installation: Jump to a heading in another file), Absolute URL (for external things)
- Must have .MD for every stage/ phase in the project, for example: DATA_PREP.md, TRAINING_INFO.md, MODEL.md, EVAL.md, ....
- All .MD must be store in a dedicated folder (even though I am not sure about this)
- AI working rules:

  - NAMING_CONVENTION.md
  - FOLDER_STRUCTURE.md
  - Run smoke test for every scripts before true running.
- Codebase_audit.md (I didn't sure what is this files, from what i heard, it related to telling the AI to align with the plan, everything we setup before it doing anything)
- For a given Task:

  1. Understand business
  2. Understand technical solution (how it done without AI)
  3. AI solution (how to apply AI into it)
  4. Data
  5. .. etc
- Deepseek and claude are notorious for hallucination and writing long scripts (make them write less).
- Commit message must be approve by asking the human first before commit.
- In ML: I as an engineer, i need to understand ML pipeline + AI working flow
- To evaluate LLM, we based on:

  - Hallucination
  - Context relevance
  - Faithfulness
- EDA can at least be done via 3 command: df.info, df.describe, df.shape
