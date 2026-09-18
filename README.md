# Departmental Procedures Assistant

A Streamlit app where you upload procedure PDFs directly in the browser and
build up a **permanent, growing knowledge base**. Answers are generated only
from what's been indexed — no outside knowledge — and every answer cites the
source document and page number.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file next to `app.py` (copy `.env.example`) and put your
   own OpenAI API key in it:

   ```
   OPENAI_API_KEY=sk-...
   ```

   > **Security note:** if you're reusing the key from your original `.env`,
   > rotate it first — a key that has been pasted into a chat or shared file
   > should be treated as compromised. Generate a fresh one at
   > https://platform.openai.com/api-keys.

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## Interface

- A proper chat interface (message bubbles, input pinned at the bottom)
  instead of a single-shot question box — your Q&A history stays visible
  as you keep asking things.
- Sources are tucked into a collapsible "📄 Sources" section under each
  answer instead of cluttering the page.
- The sidebar shows document/chunk counts as at-a-glance stats, a
  **Clear chat** button (wipes the visible conversation only — the
  knowledge base is untouched), and the upload field clears itself after
  a successful add.
- A light custom theme lives in `.streamlit/config.toml` — change
  `primaryColor` there to any hex color to match your branding.

## How it works

1. **Upload** one or more PDF procedure documents in the sidebar and click
   **Add to knowledge base**. Each PDF is split into ~300-word chunks,
   embedded with `text-embedding-3-small`, and stored in a Chroma database
   on disk, in a `chroma_db` folder created next to `app.py`.
2. **It stays indexed.** Because the knowledge base lives on disk (not just
   in memory), closing the app and running `streamlit run app.py` again
   does *not* require re-uploading anything — you can go straight to asking
   questions.
3. **Add more documents any time** to grow the knowledge base. Uploading a
   file with a name that's already indexed refreshes its chunks (handy if
   you've revised a procedure); a new filename is simply added alongside
   everything already there.
4. The sidebar shows a document/chunk count but doesn't list individual
   filenames. To remove everything, use the "Danger zone" → **Clear entire
   knowledge base** (requires an explicit confirmation checkbox, since this
   deletes permanently).
5. **Ask** a question in the main panel. The app retrieves the 5 most
   relevant chunks from the whole knowledge base and asks `gpt-4.1-mini` to
   answer using *only* that context, citing source + page. If nothing
   relevant is found, it says so rather than guessing.

## Notes

- Only text-based PDFs are supported out of the box. Scanned/image-only
  PDFs (no text layer) will show a warning — those would need OCR first.
- The `chroma_db` folder **is your knowledge base** — don't delete it
  unless you mean to, and if you move the app to a different computer or
  folder, copy `chroma_db` along with it to keep everything you've indexed.
  It's worth backing that folder up periodically.
- If this app is ever run somewhere multiple people can open it at once,
  everyone shares the same knowledge base — anything one person adds or
  removes is visible to everyone. Each person's question/answer history
  in the main panel, though, is private to their own browser session.
- Don't commit `chroma_db` or `.env` to version control — both can contain
  sensitive/proprietary content and secrets respectively. Add them to
  `.gitignore` if this project is ever put under git.
