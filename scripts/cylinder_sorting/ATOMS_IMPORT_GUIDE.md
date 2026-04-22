# ATOMS import playbook — `SYSTEM_REQUIREMENTS.md` → live project

Use this with the ATOMS MCP in Cursor (or any MCP client) to recreate the spec as a managed project with traceability. Source of truth remains [`SYSTEM_REQUIREMENTS.md`](./SYSTEM_REQUIREMENTS.md).

---

## 1. Before you start

| Step | Action |
|------|--------|
| 1 | Create the project in [ATOMS](https://atoms.tech) (MCP has no `create_project`). |
| 2 | Copy **project_id** from the app URL or from `atoms_list_projects`. |
| 3 | Prefer the **ACT Color Sorting Arm** project for the full spec; use **Default Project** only for quick experiments. |

Keep a small **ID map** as you go: `SYS-001` → `REQ-001`, `PERC-001` → `REQ-00?`, `TC-SYS-001` → `TC-001`, … ATOMS generates its own ids (`REQ-***`, `TC-***`); your spec ids live in **titles** so humans can search.

---

## 2. What we map into ATOMS

Every normative section of the spec maps to ATOMS items. Nothing in the table below is optional — it is the complete traceability payload.

| What | Source in `SYSTEM_REQUIREMENTS.md` | ATOMS representation |
|------|-----------------------------------|----------------------|
| **System requirements** | §3.2 (SYS-*) | `requirement` items, domain `system (l0)` |
| **Subsystem requirements** | §4.2 PERC · §5.3 DEC · §6.2 ACT · §7.4 TRAIN | `requirement` items, domain per subsystem |
| **Interface requirements** | §8 (IF-*) | `requirement` items, domain `interfaces` |
| **Hierarchy (parent traces)** | **Parent** column on subsystem tables | `child` links: system REQ → subsystem REQ |
| **Test cases** | §9 (TC-SYS-*, TC-PERC-*, …) | `test-case` items, domain matching their traced requirement |
| **Test → requirement traces** | §9 **Traces to** column + §10 matrix | `verifies` links: TC → REQ |
| **Test results** | §10 Status column | `atoms_record_test_result` per TC |

### What to skip

| Skip | Why |
|------|-----|
| ASCII / fenced diagrams (§3.1, §5.2, training flow) | ATOMS has trace/coverage/graph views that rebuild visuals from the linked data — no need to paste diagram text as fake requirements |
| Narrative sections (problem statement, phases, revision history) | Not normative; leave in GitHub markdown only |

---

## 3. Domains

Create all six domains **before any items**. ATOMS normalizes names to lowercase — use these exact strings in all `domains` fields:

| Domain name | Maps to | Spec section |
|-------------|---------|--------------|
| `system (l0)` | SYS-* | §3.2 |
| `perception` | PERC-* | §4.2 |
| `decision` | DEC-* | §5.3 |
| `actuation` | ACT-* | §6.2 |
| `training` | TRAIN-* | §7.4 |
| `interfaces` | IF-* | §8 |

---

## 4. Spec → ATOMS mapping

| Spec concept | ATOMS field / action |
|--------------|----------------------|
| Requirement text (EARS) | `body` via `atoms_update_item` immediately after create |
| Requirement title | `title` — prefix with spec id: `SYS-001 — …`, `IF-007 — …` |
| Domain bucket | `domains: ["perception"]` etc. |
| **Parent column** (subsystem → SYS) | `atoms_link_items`: `from_id` = **system** REQ, `to_id` = **subsystem** REQ, `type: child` |
| IF-* **parent** (interface → subsystem) | same `child` link pattern: `from_id` = subsystem REQ that owns the interface, `to_id` = IF REQ |
| **Traces to** column (TC → REQ) | `atoms_link_items`: `from_id` = TC, `to_id` = REQ, `type: verifies` |
| Test result | `atoms_record_test_result`: `result: passed / failed / blocked / not-run` |

**Verified behaviors (from pre-flight):**

- **`body` is never set on create.** Call `atoms_update_item` with `body` as a mandatory second step — skipping it leaves every item empty in the UI.
- `child` direction is **parent → child** (`from_id` = parent, `to_id` = child). The parent gets `children`, the child gets `parents`.
- Multi-parent rows (e.g. PERC-005 → SYS-001 and SYS-004) need **two separate** `child` link calls — confirmed working.
- Domain names are stored and matched lowercase.

---

## 5. Recommended import order

**Step 0 — Domains**
Create all six domains via `atoms_create_domain` before any items.

**Step 1 — System requirements (§3.2)**
SYS-001 through SYS-006. Domain: `system (l0)`.

**Step 2 — Subsystem requirements (§4.2, 5.3, 6.2, 7.4)**
PERC-*, DEC-*, ACT-*, TRAIN-*. Tag each with its matching domain. One subsystem at a time to keep rollback simple.

**Step 3 — Interface requirements (§8)**
IF-001, IF-002, IF-004 through IF-011. Domain: `interfaces`. (IF-003 was removed — it covered the GPIO interface, which is no longer in scope.) These are peer-level to subsystem requirements — import them in this step, not as an afterthought.

**Step 4 — Hierarchy links**
For every row with a **Parent** column pointing to a SYS-* id: add a `child` link from the parent system REQ to the subsystem REQ. For IF-* items, link to whichever subsystem owns that interface boundary (e.g. IF-002 → PERC-* parent, IF-007 → DEC-* parent).

**Step 5 — Test cases (§9)**
TC-SYS-*, TC-PERC-*, TC-DEC-*, TC-ACT-*, TC-TRAIN-*. Domain matching the subsystem they test.

**Step 6 — Verification links**
Each TC `verifies` the requirement(s) listed in the **Traces to** column. Multi-requirement TCs need one `verifies` link per REQ.

**Step 7 — Audit**
Run `atoms_get_coverage` — every requirement should show either `verified_by` a TC or be deliberately noted as inspection-only. Run `atoms_trace` on a sample SYS item to confirm the full chain (system → subsystem + interface → TC) is visible.

---

## 6. Tool cheat sheet

| Tool | Required args | Notes |
|------|--------------|-------|
| `atoms_create_item` | `project_id`, `type`, `title` | **Do not pass `body` here — it will not appear.** Use only `type` and `title` on create. |
| `atoms_update_item` | `project_id`, `item_id` | **Always call this immediately after create to set `body`.** Also sets `domains`, `level`, `summary`. |
| `atoms_link_items` | `project_id`, `from_id`, `to_id`, `type`, `action` | `type`: `child / parent / related / verifies / verified_by`. `action`: `add / remove` |
| `atoms_create_domain` | `project_id`, `name` | Create before items |
| `atoms_list_items` | `project_id` | Filter by `type`, `domain`, `level` |
| `atoms_get_item` | `project_id`, `item_id` | Returns full relationships and body |
| `atoms_get_coverage` | `project_id` | Shows uncovered requirements |
| `atoms_trace` | `project_id`, `item_id`, `direction`, `depth` | Blast radius / traceability view |
| `atoms_export_mermaid` | `project_id`, `item_id` | Graph output for slides |
| `atoms_record_test_result` | `project_id`, `item_id`, `result` | `result`: `passed / failed / blocked / not-run` |
| `atoms_bulk_import` | `project_id`, `items: [{type, title, body?, domains?}]` | 1–100 items per call |
| `atoms_delete_item` | `project_id`, `item_id` | No MCP restore — deletion is permanent |

---

## 7. Automation note

The import order above is designed for manual or semi-scripted MCP sessions. A parser that reads markdown tables and calls `atoms_bulk_import` is feasible — the bridge for the demo (ATOMS → `sort_config.json`) uses `atoms_list_items` to read requirement state, which also allows the bridge to be fully automated without Cursor involvement once ATOMS API credentials are available on the Jetson.

---

## 8. Pre-flight checklist (run on Default Project first)

All 12 checks verified on Default Project (`946af155-b0ef-4dfc-bcb9-0edc3e5d40d6`) — confirmed passing before the ACT Color Sorting Bot import.

| # | Test | Status |
|---|------|--------|
| 1 | `create` → `update` with `body` — full text persists in UI | PASS |
| 2 | `child` direction: parent `from_id`, child `to_id` — `parents`/`children` correct | PASS |
| 3 | `verifies`: TC `from_id`, REQ `to_id` — `verified_by` on REQ | PASS |
| 4 | Domains: create + tag — names normalize lowercase | PASS |
| 5 | Multi-parent: two `child` links to same subsystem REQ | PASS |
| 6 | `atoms_list_items` with type/domain filter | PASS |
| 7 | `atoms_get_coverage` + `atoms_trace` output readable | PASS |
| 8 | `atoms_export_mermaid` — Mermaid graph output usable | PASS |
| 9 | `atoms_record_test_result` — stored in API (`latest_result` field) | PASS |
| 10 | `atoms_bulk_import` with body | PASS |
| 11 | `atoms_delete_item` — clean delete confirmed | PASS |
| 12 | Special characters in body (`\|`, backticks, quotes, angle brackets) | PASS |

---

## 9. Quick reference — project IDs

| Project | `project_id` |
|---------|--------------|
| ACT Color Sorting Arm | `95fa5a31-7beb-4629-80d5-bb2a8d07c3e2` |
| ACT Color Sorting Bot | `60da9f5b-b659-4827-8aca-9bbdb8ba7b29` |
| Default Project | `946af155-b0ef-4dfc-bcb9-0edc3e5d40d6` |
