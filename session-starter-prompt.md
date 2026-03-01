# Session-Starter Prompt: ICAIF Paper — Final Pre-Submission Pass

Paste everything below the line into a fresh conversation, along with your paper files.

---

I'm preparing an ICAIF submission and need a structured final pre-submission pass. The paper has been through three rounds of 4-reviewer panel review already, with diminishing returns each round. I need targeted work now, not another broad review.

Please execute the following steps in order:

## Step 1: LaTeX Compilation Sanity Check

Compile the paper and report any issues with:
- Broken or unresolved `\ref{}`, `\label{}`, or `\cite{}` commands
- Orphaned footnotes or dangling cross-references
- Bibliography entries that don't resolve (pay special attention to the TOST citation and the Barnett citation)
- The `app:optima` appendix — confirm it compiles and is properly referenced from the main text
- Any LaTeX warnings (overfull hboxes, missing figures, etc.)

List every issue found. Do not proceed to Step 2 until this is clean.

## Step 2: Page Budget Check

Verify the compiled PDF fits ICAIF's sigconf format page limits (10 pages for the main body, excluding references and appendices). Report:
- Current main-body page count
- Current total page count (with appendices)
- Whether any content needs to be cut or moved to appendices to comply
- The paper currently has ~1,674 lines and 10 appendices — flag if this is excessive

If the paper exceeds limits, recommend specific cuts before proceeding.

## Step 3: Adversarial Single-Reviewer Pass

Act as a hostile finance-domain reviewer (simulating a skeptical "Reviewer 2") and identify the single strongest objection that could be raised against this paper. Focus especially on:
- **The VaR disconnect**: Is the connection between the causal regime framework and VaR application fully justified, or is there a logical gap?
- **The p=0.063 result**: Is there any decimal-unit ambiguity or presentation issue that weakens this finding? Is the statistical interpretation defensible?

For each issue found, provide:
1. The objection as a reviewer would phrase it
2. Where in the paper the weakness lives (section + line range)
3. A concrete fix (revised text or structural change)

## Step 4: Summary and Readiness Assessment

Give a go/no-go recommendation for submission. If "go," note any optional improvements. If "no-go," list the blocking issues in priority order.
