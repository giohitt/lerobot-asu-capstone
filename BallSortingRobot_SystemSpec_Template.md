# SYSTEM SPECIFICATION — Ball Sorting Robot

**NVIDIA Jetson AGX Orin 32GB + Lebrot Advanced Robot Arm**
*Systems Engineering Course — AI-Enabled Robotics*

---

| Field | Value |
|-------|-------|
| **Version** | 2.0 DRAFT |
| **Date** | [DATE] |
| **Author(s)** | [STUDENT NAMES] |
| **Course** | [COURSE NAME / NUMBER] |
| **Instructor** | [INSTRUCTOR NAME] |
| **Status** | DRAFT — For Review |

---

## Table of Contents

**Part I — Course Overview**
1. [Course Structure & Module Map](#1-course-structure--module-map)

**Part II — System Specification (Module 1 Deliverable)**
2. [Problem Statement](#2-problem-statement)
3. [System Architecture (L0)](#3-system-architecture-l0--system-level)
4. [Perception Subsystem (L1)](#4-perception-subsystem-l1)
5. [Decision Subsystem (L1)](#5-decision-subsystem-l1)
6. [Actuation Subsystem (L1)](#6-actuation-subsystem-l1)
7. [Interface Specifications](#7-interface-specifications)
8. [Verification — Test Cases](#8-verification--test-cases)
9. [Traceability Matrix](#9-traceability-matrix)

**Part III — Implementation Modules**
10. [Module 2 — Traditional Implementation](#10-module-2--traditional-implementation-yolo--ros2--state-machine)
11. [Module 3 — MCP Natural Language Control](#11-module-3--mcp-natural-language-control)
12. [Module 4 — Vision-Language-Action Model](#12-module-4--vision-language-action-model-smolvla)
13. [Module Comparison & Reflection](#13-module-comparison--reflection)

**Part IV — Administration**
14. [Revision History](#14-revision-history)
15. [Approval](#15-approval)

---

# PART I — COURSE OVERVIEW

## 1. Course Structure & Module Map

This course teaches systems engineering for AI-enabled robotics through **one specification and four implementation approaches**. Students write the system specification once (Module 1), then implement the same "pick up blue balls" task three different ways (Modules 2–4). Each module reveals different engineering trade-offs.

### 1.1 The Big Idea

> **The specification doesn't change. The implementation does. That's the point of systems engineering.**

All four modules use the same requirements (SYS-001 through SYS-006), the same test cases (Section 8), and the same success criteria. What changes is *how* the system is built — and students learn why some approaches work better than others by measuring each against the same acceptance criteria.

### 1.2 Module Overview

| Module | Title | What Students Do | Key Skill | Time |
|--------|-------|------------------|-----------|------|
| **1** | **Specification** | Write system/subsystem/component requirements, draw architecture in ATOMS Whiteboard, define test cases | Systems engineering, V-model, requirements writing | 2–3 weeks |
| **2** | **Traditional Implementation** | Train YOLO object detection, generate ROS2 code from spec, integrate on Jetson | AI training, code generation from specs, integration testing | 3–4 weeks |
| **3** | **MCP Natural Language Control** | Set up Robot MCP Server, control arm via Claude with natural language commands | MCP protocol, LLM-as-controller, failure analysis | 1–2 weeks |
| **4** | **Vision-Language-Action Model** | Teleoperate arm for demonstrations, fine-tune SmolVLA, deploy with natural language task prompt | VLA models, imitation learning, fine-tuning | 2–3 weeks |

### 1.3 Learning Progression

```
MODULE 1: SPECIFY                MODULE 2: BUILD              MODULE 3: TALK             MODULE 4: TEACH
─────────────────                ────────────────             ──────────────             ───────────────
Write requirements     →    Train YOLO + code gen    →    "Pick up blue ball"    →    Demonstrate 50x
Draw architecture      →    Integrate ROS2 nodes     →    Claude sends joints    →    Fine-tune SmolVLA
Define test cases      →    Run test cases           →    Measure failure rate   →    "Pick up blue ball"
                                                                                     (but now it works)
OUTCOME:                   OUTCOME:                    OUTCOME:                   OUTCOME:
"What to build"            Reliable but slow           Fast to try, unreliable    Natural language →
                           to develop                                             trained behavior

TEACHES:                   TEACHES:                    TEACHES:                   TEACHES:
Requirements matter        Specs drive code gen        LLMs aren't magic          Data > prompts for
                                                       for physical tasks         physical control
```

### 1.4 Hardware & Software Requirements

| Component | Purpose | Modules |
|-----------|---------|---------|
| NVIDIA Jetson AGX Orin 32GB | Compute platform | All |
| Lebrot Advanced Robot Arm | Manipulation | All |
| USB/CSI Camera | Visual perception | All |
| ATOMS Platform | Requirements management & whiteboard | 1 |
| ROS2 Humble | Robotics middleware | 2, 4 |
| Ultralytics YOLOv8 | Object detection training | 2 |
| Claude Desktop + MCP | Natural language robot control | 3 |
| LeRobot + SmolVLA | Vision-language-action model | 4 |
| Teleoperation device | Record demonstrations | 4 |

### 1.5 Grading Rubric (Suggested)

| Component | Weight | Criteria |
|-----------|--------|----------|
| Module 1 — Specification Document | 30% | Completeness, traceability, EARS format, testability |
| Module 2 — Traditional Implementation | 25% | Detection accuracy, test case pass rate, code traceability |
| Module 3 — MCP Control & Analysis | 15% | Setup, experimentation, failure analysis writeup |
| Module 4 — VLA Fine-Tuning | 20% | Data collection quality, training results, success rate |
| Final Reflection — Module Comparison | 10% | Quantitative comparison, engineering trade-off analysis |

---

# PART II — SYSTEM SPECIFICATION (Module 1 Deliverable)

> **🎯 MODULE 1 STARTS HERE**
>
> Students complete Sections 2–9 as their Module 1 deliverable. This specification drives all subsequent modules. No implementation begins until the spec is reviewed and approved.

---

## 2. Problem Statement

This section defines the problem the system must solve. Students should describe the task in plain language before writing any requirements.

### 2.1 Mission Statement

Design and build an autonomous robotic system that visually identifies blue balls in a workspace, picks them up using a robot arm, and places them into a designated bin. The system shall operate without human intervention after initialization.

### 2.2 Operational Context

- **Environment:** Indoor lab bench, controlled lighting (300–500 lux)
- **Workspace:** Approximately 60cm × 40cm flat surface
- **Objects:** Blue balls (25–40mm diameter), with possible distractor objects (red balls, cubes)
- **Hardware:** NVIDIA Jetson AGX Orin 32GB, Lebrot Advanced robot arm, USB/CSI camera
- **Operator:** Student initiates system via command; no further human input required

### 2.3 Success Criteria

| ID | Criterion | Threshold |
|----|-----------|-----------|
| SC-001 | System picks and places blue balls into bin | ≥80% success rate over 10 trials |
| SC-002 | System ignores non-blue objects | 0 false picks per trial |
| SC-003 | Full cycle completes autonomously | No human intervention after start command |
| SC-004 | End-to-end cycle time per ball | ≤30 seconds per ball |

---

## 3. System Architecture (L0 — System Level)

The system is decomposed into three primary subsystems following the **Perception–Decision–Action** pattern common in autonomous systems. This is the top-level view that would appear in the ATOMS Canvas at Level 0.

### 3.1 System Block Diagram

> *Students: Draw this in the ATOMS Whiteboard. Each box becomes a subsystem. Each arrow becomes an interface with requirements.*

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                BALL SORTING ROBOT (L0)                  │
                    ├─────────────────────────────────────────────────────────┤
                    │                                                         │
                    │  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
                    │  │PERCEPTION│──→│ DECISION │──→│ACTUATION │           │
                    │  │          │   │          │   │          │           │
                    │  │ Camera   │   │ State    │   │ Arm      │           │
                    │  │ Detect   │   │ Machine  │   │ Gripper  │           │
                    │  │ Localize │   │ Plan     │   │ Safety   │           │
                    │  └──────────┘   └──────────┘   └──────────┘           │
                    │                                                         │
                    └─────────────────────────────────────────────────────────┘
```

### 3.2 System-Level Requirements

| ID | Requirement (EARS Format) | Rationale | Verification |
|----|---------------------------|-----------|--------------|
| SYS-001 | The system SHALL identify blue balls within the workspace with ≥90% detection accuracy. | Core mission capability | TC-SYS-001 |
| SYS-002 | The system SHALL pick up identified blue balls and place them in the designated bin. | Core mission capability | TC-SYS-002 |
| SYS-003 | The system SHALL complete one pick-and-place cycle in ≤30 seconds. | Operational efficiency | TC-SYS-003 |
| SYS-004 | The system SHALL NOT pick up objects that are not blue balls. | Selectivity requirement | TC-SYS-004 |
| SYS-005 | The system SHALL operate autonomously after receiving a start command. | Autonomy requirement | TC-SYS-005 |
| SYS-006 | The system SHALL stop all motion within 500ms when emergency stop is triggered. | Safety requirement | TC-SYS-006 |

---

## 4. Perception Subsystem (L1)

The Perception Subsystem is responsible for visual sensing, object detection, and spatial localization. **This is where AI model training fits into the systems engineering process (Module 2), or where the VLA model handles everything end-to-end (Module 4).**

### 4.1 Subsystem Decomposition

| Component | Function | Module 2 Technology | Module 4 Technology |
|-----------|----------|---------------------|---------------------|
| Camera Module | Capture video frames of workspace | USB/CSI camera on Jetson | Same |
| Object Detection Model | Identify and locate blue balls in frame | YOLOv8-nano (trained by students) | SmolVLA (handles detection + action together) |
| Spatial Localizer | Convert pixel coordinates to 3D workspace position | Camera calibration + depth estimation | Learned implicitly by VLA model |

### 4.2 Perception Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| PERC-001 | The Camera Module SHALL capture frames at ≥15 fps at minimum 640×480 resolution. | SYS-001 | TC-PERC-001 |
| PERC-002 | The Object Detection Model SHALL detect blue balls with ≥90% mAP@0.5 on the validation dataset. | SYS-001 | TC-PERC-002 |
| PERC-003 | The Object Detection Model SHALL run inference in ≤60ms per frame on Jetson AGX. | SYS-003 | TC-PERC-003 |
| PERC-004 | The Object Detection Model SHALL NOT detect non-blue objects as blue balls (false positive rate ≤5%). | SYS-004 | TC-PERC-004 |
| PERC-005 | The Spatial Localizer SHALL estimate ball center position with ±10mm accuracy. | SYS-002 | TC-PERC-005 |
| PERC-006 | The system SHALL operate under lab lighting conditions (300–500 lux). | SYS-001 | TC-PERC-006 |

### 4.3 AI Model Training Pipeline (Module 2)

This section documents how the AI model (a component) is implemented via traditional training. Training a model is analogous to writing code — it is the implementation step that fulfills perception requirements. The trained model is a deliverable artifact, just like compiled code.

> **🔑 KEY INSIGHT: The AI model is a COMPONENT, not magic. It has requirements, is implemented (trained), tested, and verified — just like any other component.**

#### 4.3.1 Data Collection Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| TRAIN-001 | Training dataset SHALL contain ≥200 images of blue balls in the target workspace. | Minimum for reliable fine-tuning of YOLOv8-nano |
| TRAIN-002 | Dataset SHALL include variation in ball position, lighting angle, and partial occlusion. | Robustness to real-world conditions |
| TRAIN-003 | Dataset SHALL include ≥50 negative images (no blue balls, or only distractor objects). | Reduces false positive rate (PERC-004) |
| TRAIN-004 | All images SHALL be annotated with bounding boxes in YOLO format. | Required input format for training pipeline |
| TRAIN-005 | Dataset SHALL be split 80% training / 20% validation. | Standard practice for model evaluation |

#### 4.3.2 Training Process

> *Students: Document each step you perform. This is your implementation record.*

1. Capture images using Jetson camera in the actual workspace
2. Label images using annotation tool (e.g., Roboflow, CVAT, or LabelImg)
3. Export annotations in YOLO format
4. Train YOLOv8-nano using Ultralytics on Jetson AGX
5. Evaluate model on validation set — record mAP@0.5
6. If mAP < 90%, iterate: collect more data, adjust augmentation, retrain
7. Export final model to TensorRT for optimized Jetson inference

#### 4.3.3 Training Results Log

> *Students: Fill in this table each time you train. This is your evidence for PERC-002.*

| Run # | Date | Dataset Size | Epochs | mAP@0.5 | Inference Time (ms) | Pass/Fail (PERC-002) |
|-------|------|-------------|--------|---------|--------------------|-----------------------|
| 1 | [DATE] | [N images] | [N] | [X.X%] | [X ms] | [PASS/FAIL] |
| 2 | [DATE] | [N images] | [N] | [X.X%] | [X ms] | [PASS/FAIL] |
| 3 | [DATE] | [N images] | [N] | [X.X%] | [X ms] | [PASS/FAIL] |

---

## 5. Decision Subsystem (L1)

The Decision Subsystem processes perception outputs and determines what action to take. It manages the operational state machine and coordinates between perception and actuation.

### 5.1 Subsystem Decomposition

| Component | Function | Module 2 Technology | Module 3/4 Technology |
|-----------|----------|---------------------|------------------------|
| State Machine Controller | Manage system operational states | Python / ROS2 node | Claude reasoning (M3) / VLA implicit (M4) |
| Target Selector | Choose which detected ball to pick next | Nearest-ball heuristic | Claude visual reasoning (M3) / VLA implicit (M4) |
| Path Planner | Compute collision-free arm trajectory | Lebrot SDK / MoveIt2 | Claude iterative commands (M3) / VLA direct actions (M4) |

### 5.2 State Machine Definition

> *Students: This state machine drives your code generation in Module 2. In Module 3, Claude follows this logic implicitly. In Module 4, the VLA model learns it from demonstrations.*

```
IDLE → [Start Command] → SCANNING → [Ball Detected] → TARGETING → [Path Planned] → PICKING
  ↑                                                                                    │
  │         ┌──────────────────────────────────────────────────────────────────────────┘
  │         │
  │         ▼
  │      PLACING → [Ball Released] → SCANNING → ... → [No Balls for 5s] → COMPLETE
  │
  └──── ERROR ← [Any Fault Detected from Any State]
```

| State | Entry Condition | Actions | Exit Condition |
|-------|-----------------|---------|----------------|
| IDLE | System powered on | Wait for start command | Start command received |
| SCANNING | Previous state complete | Capture frame, run detection | Ball detected OR timeout (5s) |
| TARGETING | Ball detected | Select nearest ball, compute arm path | Path computed |
| PICKING | Path ready | Move arm to ball, close gripper | Gripper force confirms grasp |
| PLACING | Ball grasped | Move arm to bin, open gripper | Gripper opened at bin position |
| ERROR | Any fault detected | Stop arm, log error, alert operator | Operator reset |
| COMPLETE | No balls for 5 seconds | Return arm to home, report results | Operator restart |

### 5.3 Decision Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| DEC-001 | The State Machine SHALL transition through defined states without skipping states. | SYS-005 | TC-DEC-001 |
| DEC-002 | The Target Selector SHALL choose the ball nearest to the arm's current position. | SYS-003 | TC-DEC-002 |
| DEC-003 | The Path Planner SHALL generate a collision-free trajectory within 2 seconds. | SYS-003 | TC-DEC-003 |
| DEC-004 | WHEN an error is detected, the system SHALL transition to ERROR state within 200ms. | SYS-006 | TC-DEC-004 |
| DEC-005 | WHEN no balls are detected for 5 consecutive seconds, the system SHALL transition to COMPLETE. | SYS-005 | TC-DEC-005 |

---

## 6. Actuation Subsystem (L1)

The Actuation Subsystem physically manipulates objects using the Lebrot Advanced robot arm and gripper.

### 6.1 Subsystem Decomposition

| Component | Function | Technology |
|-----------|----------|------------|
| Arm Controller | Execute joint-level trajectory commands | Lebrot SDK via serial/USB |
| Gripper Controller | Open/close gripper, detect grasp | Lebrot gripper module |
| Safety Monitor | Enforce joint limits, detect collisions, e-stop | Watchdog timer + limit checks |

### 6.2 Actuation Requirements

| ID | Requirement | Parent | Verification |
|----|-------------|--------|--------------|
| ACT-001 | The Arm Controller SHALL position the gripper within ±5mm of the target position. | SYS-002 | TC-ACT-001 |
| ACT-002 | The Gripper SHALL close with sufficient force to hold a 25–40mm ball without crushing it. | SYS-002 | TC-ACT-002 |
| ACT-003 | The Arm SHALL return to home position after placing a ball. | SYS-005 | TC-ACT-003 |
| ACT-004 | The Safety Monitor SHALL halt arm motion if joint limits are exceeded. | SYS-006 | TC-ACT-004 |
| ACT-005 | WHEN emergency stop is triggered, all motors SHALL de-energize within 500ms. | SYS-006 | TC-ACT-005 |

---

## 7. Interface Specifications

**Interfaces are where systems fail.** Each arrow in the ATOMS Whiteboard diagram represents a data flow between subsystems. Each interface has its own requirements.

> **In ATOMS Whiteboard: Link these requirements to the ARROWS between boxes, not to the boxes themselves.**

| ID | Interface (Arrow) | Data Format | Requirement |
|----|-------------------|-------------|-------------|
| IF-001 | Camera → Object Detection | Video frames (BGR, 640×480, 15fps) | Frame delivery latency SHALL be ≤20ms |
| IF-002 | Object Detection → Target Selector | List of detections: `[{class, confidence, bbox_xywh}]` | Detection list SHALL update every frame |
| IF-003 | Target Selector → Path Planner | Target position: `{x, y, z}` in workspace coords (mm) | Position SHALL have ±10mm accuracy |
| IF-004 | Path Planner → Arm Controller | Joint trajectory: `[{joint_angles[], timestamp}]` | Trajectory SHALL be collision-checked before execution |
| IF-005 | State Machine → Gripper Controller | Command: `OPEN | CLOSE` | Command acknowledgment within 100ms |
| IF-006 | Gripper Controller → State Machine | Status: `{grasped: bool, force_N: float}` | Status update within 50ms of state change |

> *Note: In Module 3 (MCP), Claude replaces IF-002 through IF-004 with its own reasoning. In Module 4 (VLA), the model replaces all interfaces — it goes directly from camera pixels to motor commands.*

---

## 8. Verification — Test Cases

Each requirement must have at least one test case. Tests trace back to requirements, completing the V-model. **These same test cases are used to evaluate ALL modules** — this is how students compare approaches quantitatively.

> *Students: Run these tests for each module and record results. The final reflection (Section 13) compares results across modules.*

### 8.1 System-Level Tests

| ID | Traces To | Procedure | Pass Criteria | M2 Result | M3 Result | M4 Result |
|----|-----------|-----------|---------------|-----------|-----------|-----------|
| TC-SYS-001 | SYS-001 | Place 10 blue balls in workspace. Run system. Count detections. | ≥9 of 10 detected | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-SYS-002 | SYS-002 | Place 5 blue balls. Run full cycle. Count successful placements in bin. | ≥4 of 5 placed | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-SYS-003 | SYS-003 | Time from detection to ball-in-bin for 5 cycles. | Mean ≤30 seconds | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-SYS-004 | SYS-004 | Place 5 blue balls + 5 red balls. Run system. | 0 red balls picked | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-SYS-005 | SYS-005 | Start system. Do not touch anything. Observe full cycle completion. | Completes without intervention | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-SYS-006 | SYS-006 | During arm motion, trigger e-stop. Measure stop time. | All motion stops ≤500ms | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |

### 8.2 Perception Tests

| ID | Traces To | Procedure | Pass Criteria | M2 Result | M3 Result | M4 Result |
|----|-----------|-----------|---------------|-----------|-----------|-----------|
| TC-PERC-001 | PERC-001 | Run camera capture node. Measure fps over 60 seconds. | ≥15 fps, ≥640×480 | [PASS/FAIL] | N/A | [PASS/FAIL] |
| TC-PERC-002 | PERC-002 | Run model on validation set (from TRAIN-005 split). | mAP@0.5 ≥90% | [PASS/FAIL] | N/A | N/A |
| TC-PERC-003 | PERC-003 | Measure inference time per frame over 100 frames. | Mean ≤60ms | [PASS/FAIL] | N/A | [PASS/FAIL] |
| TC-PERC-004 | PERC-004 | Present 20 red balls and 10 cubes. Run detection. | ≤1 false positive | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-PERC-005 | PERC-005 | Place ball at 5 known positions. Compare estimated vs actual position. | Error ≤10mm for all 5 | [PASS/FAIL] | N/A | N/A |

### 8.3 Decision & Actuation Tests

| ID | Traces To | Procedure | Pass Criteria | M2 Result | M3 Result | M4 Result |
|----|-----------|-----------|---------------|-----------|-----------|-----------|
| TC-DEC-001 | DEC-001 | Log state transitions for 5 cycles. Verify sequence. | No skipped states | [PASS/FAIL] | N/A | N/A |
| TC-DEC-003 | DEC-003 | Measure path planning time for 10 targets. | All ≤2 seconds | [PASS/FAIL] | [PASS/FAIL] | N/A |
| TC-ACT-001 | ACT-001 | Command arm to 5 known positions. Measure error. | All ≤5mm error | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-ACT-002 | ACT-002 | Grip 10 balls. Count drops during transport. | ≤1 drop in 10 | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |
| TC-ACT-005 | ACT-005 | Trigger e-stop during motion. Measure with high-speed timer. | Motor stop ≤500ms | [PASS/FAIL] | [PASS/FAIL] | [PASS/FAIL] |

---

## 9. Traceability Matrix

This matrix shows that every requirement traces to at least one test, and every test traces back to a requirement. In ATOMS, this is generated automatically from linked items.

| Requirement | Subsystem | Test Case(s) | Status |
|-------------|-----------|-------------|--------|
| SYS-001 | System | TC-SYS-001, TC-PERC-002 | [NOT TESTED] |
| SYS-002 | System | TC-SYS-002, TC-ACT-001 | [NOT TESTED] |
| SYS-003 | System | TC-SYS-003, TC-DEC-003 | [NOT TESTED] |
| SYS-004 | System | TC-SYS-004, TC-PERC-004 | [NOT TESTED] |
| SYS-005 | System | TC-SYS-005, TC-DEC-001 | [NOT TESTED] |
| SYS-006 | System | TC-SYS-006, TC-ACT-005 | [NOT TESTED] |
| PERC-001 | Perception | TC-PERC-001 | [NOT TESTED] |
| PERC-002 | Perception | TC-PERC-002 | [NOT TESTED] |
| PERC-003 | Perception | TC-PERC-003 | [NOT TESTED] |
| PERC-004 | Perception | TC-PERC-004 | [NOT TESTED] |
| PERC-005 | Perception | TC-PERC-005 | [NOT TESTED] |
| DEC-001 | Decision | TC-DEC-001 | [NOT TESTED] |
| DEC-003 | Decision | TC-DEC-003 | [NOT TESTED] |
| ACT-001 | Actuation | TC-ACT-001 | [NOT TESTED] |
| ACT-002 | Actuation | TC-ACT-002 | [NOT TESTED] |
| ACT-005 | Actuation | TC-ACT-005 | [NOT TESTED] |

> **🎯 MODULE 1 DELIVERABLE ENDS HERE.** Students submit Sections 2–9 for review before proceeding to implementation modules.

---

# PART III — IMPLEMENTATION MODULES

---

## 10. Module 2 — Traditional Implementation (YOLO + ROS2 + State Machine)

### 10.1 Overview

| Aspect | Detail |
|--------|--------|
| **Approach** | Train a YOLO object detection model, generate ROS2 code from the specification, integrate components on Jetson |
| **Key Skill** | AI training as component implementation, spec-driven code generation |
| **Expected Reliability** | High (~80%+ success rate) |
| **Development Time** | 3–4 weeks |
| **Coding Required** | Yes — AI-assisted code generation from spec, plus integration |

### 10.2 What Gets Generated vs. What Gets Trained

| Component | Implementation Method | Input to AI Code Generator |
|-----------|----------------------|----------------------------|
| State Machine Controller | **CODE GENERATION** — Feed Section 5.2 to Claude/Copilot | State machine table + transition conditions |
| Camera Module | **CODE GENERATION** — Standard ROS2 camera node | PERC-001 frame rate and resolution requirements |
| Object Detection Model | **🔴 AI TRAINING** — Students collect data and train YOLOv8 | TRAIN-001 through TRAIN-005 define the training process |
| Spatial Localizer | **CODE GENERATION** — Coordinate transform math | PERC-005 accuracy requirement + camera calibration data |
| Path Planner | **CODE GENERATION** — Use Lebrot SDK / MoveIt2 API | DEC-003 timing requirement + ACT-001 accuracy requirement |
| Gripper Controller | **CODE GENERATION** — Serial command interface | ACT-002 force requirement + IF-005/IF-006 interface spec |
| Safety Monitor | **CODE GENERATION** — Watchdog + limit checks | SYS-006 + ACT-004 + ACT-005 safety requirements |

### 10.3 AI Prompt Template for Code Generation

> *Students: Use this template when prompting Claude or Copilot to generate code from your requirements.*

```
"Generate a Python ROS2 node for the [COMPONENT NAME] component of a ball sorting robot.

The component must satisfy the following requirements:
[PASTE REQUIREMENTS TABLE]

The interfaces are:
[PASTE RELEVANT IF- REQUIREMENTS]

Use the following state machine:
[PASTE STATE TABLE]

Target platform is NVIDIA Jetson AGX Orin running Ubuntu 22.04 with ROS2 Humble."
```

### 10.4 Module 2 Workflow

1. **TRAIN** — Collect data, label, train YOLO model (Section 4.3) — fulfills PERC-002
2. **GENERATE** — Feed requirements + state machine to AI coding tool → get ROS2 nodes
3. **INTEGRATE** — Connect nodes via ROS2 topics matching interface specs (Section 7)
4. **TEST** — Execute test cases from Section 8, record results in M2 columns
5. **VERIFY** — Update traceability matrix — every requirement has a passing test

### 10.5 Module 2 Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | Trained YOLO model + training results log (Section 4.3.3) | Model weights + filled table |
| 2 | Generated ROS2 nodes with requirement traceability comments | Python source files |
| 3 | Integration test results | Filled test case tables (M2 column) |
| 4 | Updated traceability matrix | Section 9 updated |

---

## 11. Module 3 — MCP Natural Language Control

### 11.1 Overview

| Aspect | Detail |
|--------|--------|
| **Approach** | Connect a Robot MCP Server to the Lebrot arm. Use Claude Desktop (or agent script) to control the robot with natural language commands. |
| **Key Skill** | MCP protocol, understanding LLM limitations in physical control |
| **Expected Reliability** | Low (~20–40% success rate) |
| **Development Time** | 1–2 weeks |
| **Coding Required** | Minimal — MCP server setup + configuration only |

### 11.2 How It Works

```
┌──────────────┐     Natural Language      ┌──────────────┐     Joint Commands    ┌──────────────┐
│              │  "Pick up the blue ball"   │              │   move_joint(3, 45°)  │              │
│    Student   │ ────────────────────────→  │    Claude    │ ──────────────────→   │  Robot Arm   │
│              │                            │  (via MCP)   │ ←──────────────────   │  (Lebrot)    │
│              │  ←────────────────────     │              │   camera_image.jpg    │              │
│              │  "I can see 2 blue balls"  │              │                       │              │
└──────────────┘                            └──────────────┘                       └──────────────┘
```

Claude receives camera images, reasons about what it sees, then sends joint-level commands to the arm through the MCP server. Each "think → look → move" cycle requires a full LLM API call.

### 11.3 Architecture: Claude as Controller

| Traditional (Module 2) | MCP (Module 3) |
|-------------------------|----------------|
| YOLO detects ball position | Claude looks at camera image and describes what it sees |
| Algorithm selects nearest ball | Claude reasons: "The closest blue ball is to the right" |
| Path planner computes trajectory | Claude sends: `move_joint(base, 30°)` then checks camera again |
| Runs at 15+ fps | Runs at ~0.1 fps (one cycle every ~10 seconds) |
| Deterministic | Non-deterministic (different reasoning each time) |

### 11.4 Setup Steps

1. Install Robot MCP Server (adapt from [IliaLarchenko/robot_MCP](https://github.com/IliaLarchenko/robot_MCP) for Lebrot arm)
2. Configure serial port and camera in `config.py`
3. Calibrate joint angle mappings for Lebrot
4. Test manual keyboard control first
5. Connect MCP server to Claude Desktop (or use `agent.py` script)
6. Start with simple commands: "Move the arm up", "Open the gripper"
7. Attempt full task: "Pick up the blue ball and put it in the bin"

### 11.5 Experiment Protocol

> *Students: Run these experiments and record data. This is your Module 3 evidence.*

| Experiment | Procedure | Record |
|------------|-----------|--------|
| EXP-MCP-001 | Give Claude 10 attempts to pick up 1 blue ball (no distractors) | Success count out of 10 |
| EXP-MCP-002 | Give Claude 5 attempts with blue + red balls mixed | Correct picks, false picks, misses |
| EXP-MCP-003 | Time a single successful pick-and-place cycle | Total time in seconds |
| EXP-MCP-004 | Count total LLM API calls per successful cycle | Number of calls, estimated token cost |
| EXP-MCP-005 | Log Claude's reasoning for 3 failed attempts | Copy full conversation transcript |

### 11.6 Expected Failure Modes

> *Students: Document which of these you observe and why.*

| Failure Mode | Why It Happens | Systems Engineering Lesson |
|--------------|----------------|----------------------------|
| Arm overshoots target | Claude can't perceive depth well from a 2D image | PERC-005 (±10mm accuracy) is hard without depth sensing |
| Gripper misaligns | Claude sends discrete commands, not smooth trajectories | IF-004 (trajectory) matters — discrete steps lose precision |
| Ball rolls away on approach | No force feedback in the LLM loop | IF-006 (gripper status) is missing from MCP control loop |
| Picks wrong object | Claude misidentifies colors in certain lighting | PERC-006 (lighting conditions) affects LLM vision too |
| Takes 2+ minutes per ball | Every movement requires a full LLM inference cycle | SYS-003 (≤30s) is nearly impossible with LLM-in-the-loop |

### 11.7 Module 3 Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | Working MCP server configuration for Lebrot arm | Config files + setup documentation |
| 2 | Experiment results (EXP-MCP-001 through EXP-MCP-005) | Filled table + conversation transcripts |
| 3 | System test results | Filled test case tables (M3 column) |
| 4 | Failure analysis writeup (1–2 pages) | Discussion of why MCP approach fails specific requirements |

---

## 12. Module 4 — Vision-Language-Action Model (SmolVLA)

### 12.1 Overview

| Aspect | Detail |
|--------|--------|
| **Approach** | Teleoperate the arm to demonstrate the task ~50 times, fine-tune SmolVLA on demonstrations, deploy model that takes natural language + camera → motor commands |
| **Key Skill** | Imitation learning, VLA fine-tuning, data-driven robotics |
| **Expected Reliability** | Medium (~40–70% success rate, improves with more demonstrations) |
| **Development Time** | 2–3 weeks |
| **Coding Required** | Minimal — LeRobot scripts for recording, training, inference |

### 12.2 How It Works

```
PHASE 1: TEACH (Teleoperation)                PHASE 2: DEPLOY (Autonomous)
─────────────────────────────                  ──────────────────────────────

┌──────────┐   Human moves    ┌──────────┐    ┌──────────┐  "Pick up blue   ┌──────────┐
│ Leader   │  ──────────────→ │ Follower │    │  Camera  │   balls and put   │ SmolVLA  │
│ Arm      │   arm through    │ Arm      │    │  Image   │   in bin"         │  Model   │
│(teleop)  │   task 50×       │(records) │    │          │ ──────────────→   │          │
└──────────┘                  └──────────┘    └──────────┘                   └────┬─────┘
                                    │                                             │
                              Records to                                   Direct motor
                              LeRobot Dataset                              commands at 50Hz
                                    │                                             │
                                    ▼                                             ▼
                              ┌──────────┐                                ┌──────────┐
                              │ Fine-tune│                                │ Robot    │
                              │ SmolVLA  │                                │ Arm      │
                              │ on Jetson│                                │ Executes │
                              └──────────┘                                └──────────┘
```

### 12.3 The Key Difference from Module 2

| Module 2 (Traditional) | Module 4 (VLA) |
|-------------------------|----------------|
| Separate models for perception, planning, control | **One model** does everything end-to-end |
| Train object detection on labeled images | Train behavior from teleoperated demonstrations |
| Code connects components via interfaces (Section 7) | No interfaces — model maps pixels → actions directly |
| Explicit state machine (Section 5.2) | State machine is learned implicitly from demos |
| Requires ROS2, path planning, gripper control code | Requires only LeRobot inference script |

### 12.4 Data Collection Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| VLA-001 | Demonstration dataset SHALL contain ≥50 successful teleoperated episodes of the complete pick-and-place task. | Minimum recommended by SmolVLA documentation for fine-tuning |
| VLA-002 | Each episode SHALL include synchronized camera frames and joint state/action data in LeRobot v3 format. | Required format for SmolVLA training pipeline |
| VLA-003 | Demonstrations SHALL include variation in ball starting position (≥5 distinct positions). | Generalization across workspace |
| VLA-004 | All episodes SHALL use the task description: "Pick up the blue ball and place it in the bin." | Natural language instruction required by VLA architecture |
| VLA-005 | Dataset SHALL be uploaded to Hugging Face Hub for reproducibility. | Standard LeRobot practice, enables collaboration |

### 12.5 Module 4 Workflow

1. **SETUP** — Install LeRobot on Jetson, configure Lebrot arm as follower, set up teleoperation device
2. **RECORD** — Teleoperate arm through 50+ successful pick-and-place demonstrations
    ```bash
    lerobot-record \
      --robot.type=lebrot_advanced \
      --robot.port=/dev/ttyACM0 \
      --dataset.single_task="Pick up the blue ball and place it in the bin." \
      --dataset.repo_id=${HF_USER}/ball_sorting_demos \
      --dataset.num_episodes=50
    ```
3. **VALIDATE** — Replay 5 episodes to verify data quality
4. **TRAIN** — Fine-tune SmolVLA on recorded dataset
    ```bash
    python lerobot/scripts/train.py \
      --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=${HF_USER}/ball_sorting_demos \
      --training.num_steps=20000
    ```
5. **DEPLOY** — Run inference with natural language prompt
    ```bash
    lerobot-record \
      --robot.type=lebrot_advanced \
      --dataset.single_task="Pick up the blue ball and place it in the bin." \
      --policy.path=${HF_USER}/ball_sorting_smolvla
    ```
6. **TEST** — Execute test cases from Section 8, record results in M4 columns

### 12.6 Training Results Log

> *Students: Fill in this table for each training run.*

| Run # | Date | Episodes | Training Steps | Task Success Rate (10 trials) | Notes |
|-------|------|----------|----------------|-------------------------------|-------|
| 1 | [DATE] | [N] | [N] | [X/10] | [Notes on failures] |
| 2 | [DATE] | [N] | [N] | [X/10] | [Notes on improvements] |
| 3 | [DATE] | [N] | [N] | [X/10] | [Notes] |

### 12.7 Expected Challenges

| Challenge | Why It Happens | Mitigation |
|-----------|----------------|------------|
| Model fails to grasp | Insufficient demonstration variety | Record more episodes with different ball positions |
| Arm trajectory is jerky | Action chunking artifacts | Increase action chunk size, add temporal smoothing |
| Model ignores language instruction | Task description not in training data | Ensure EVERY episode has the correct task string (VLA-004) |
| Works in one position only | All demos started from same ball location | Vary starting conditions per VLA-003 |
| Training takes too long on Jetson | SmolVLA fine-tuning is GPU-intensive | Use `train_expert_only=true` to freeze VLM backbone |

### 12.8 Module 4 Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | Recorded demonstration dataset (≥50 episodes) | LeRobot dataset on Hugging Face Hub |
| 2 | Fine-tuned SmolVLA model | Model weights on Hugging Face Hub |
| 3 | Training results log (Section 12.6) | Filled table |
| 4 | System test results | Filled test case tables (M4 column) |
| 5 | Video of 3 successful autonomous pick-and-place cycles | MP4 recording |

---

## 13. Module Comparison & Reflection

### 13.1 Quantitative Comparison

> *Students: Fill in this table after completing all modules. Use data from Section 8 test results.*

| Metric | Module 2 (Traditional) | Module 3 (MCP) | Module 4 (VLA) |
|--------|------------------------|----------------|----------------|
| **Pick success rate** (TC-SYS-002) | [X/5] | [X/5] | [X/5] |
| **False pick rate** (TC-SYS-004) | [X red picked] | [X red picked] | [X red picked] |
| **Cycle time per ball** (TC-SYS-003) | [X seconds] | [X seconds] | [X seconds] |
| **E-stop response** (TC-SYS-006) | [X ms] | [X ms] | [X ms] |
| **Development time** | [X hours] | [X hours] | [X hours] |
| **Lines of code written** | [N] | [N] | [N] |
| **Cost per run** | ~$0 (local) | ~$[X] (API tokens) | ~$0 (local) |
| **Requires training data** | Yes (200 images) | No | Yes (50 demonstrations) |

### 13.2 Reflection Questions

> *Students: Write 1–2 paragraphs for each question.*

1. **Which module achieved the highest success rate? Why?** Consider the engineering trade-offs that led to this result.

2. **Which requirements from Sections 4–6 were impossible to meet in Module 3 (MCP)? What does this teach you about using LLMs for physical control?**

3. **Module 2 required explicit interfaces (Section 7). Module 4 learned them implicitly. What are the safety implications of implicit vs. explicit interfaces in a real product?**

4. **If you were building a medical device that sorts pills by color, which module's approach would you choose and why? How would regulatory requirements (FDA, ISO 26262) influence your decision?**

5. **The specification (Module 1) was the same across all modules. Did writing the spec first change how you approached Modules 2–4? Would you have done anything differently without the spec?**

### 13.3 The Big Takeaway

```
MODULE 1 (Spec)          → Defines WHAT the system must do
MODULE 2 (Traditional)   → Shows HOW to build it reliably with explicit engineering
MODULE 3 (MCP)           → Shows WHY natural language alone isn't enough for physical systems
MODULE 4 (VLA)           → Shows WHERE the field is going — but still needs disciplined data collection

The specification doesn't go away. It changes what it drives:
  Module 2: Spec drives CODE GENERATION
  Module 3: Spec drives EVALUATION of Claude's attempts
  Module 4: Spec drives DATA COLLECTION discipline and ACCEPTANCE CRITERIA
```

---

# PART IV — ADMINISTRATION

## 14. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | [DATE] | [STUDENT] | Module 1 deliverable — system specification |
| 1.1 | [DATE] | [STUDENT] | Module 2 results — YOLO training + ROS2 integration |
| 1.2 | [DATE] | [STUDENT] | Module 3 results — MCP experiments + failure analysis |
| 1.3 | [DATE] | [STUDENT] | Module 4 results — SmolVLA fine-tuning + deployment |
| 2.0 | [DATE] | [STUDENT] | Final submission — module comparison + reflection |

---

## 15. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Student Lead | [NAME] | [DATE] | |
| Team Member | [NAME] | [DATE] | |
| Team Member | [NAME] | [DATE] | |
| Instructor | [NAME] | [DATE] | |
