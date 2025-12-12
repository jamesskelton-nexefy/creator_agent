/**
 * Architect Agent
 *
 * Builds the COMPLETE STRUCTURE of training courses (Levels 2 through Content Block).
 * Creates the full skeleton from Sections down to Content Blocks (empty shells).
 * Presents plan to user for approval before building.
 * Does NOT write the actual content - that's the Writer's job.
 *
 * Tools:
 * - requestPlanApproval - Present plan to user and get approval before building
 * - requestEditMode, releaseEditMode - Edit lock management
 * - createNode - Create structural nodes (sections, topics, content blocks, etc.)
 * - getProjectHierarchyInfo - Understand hierarchy levels and coding (CALL FIRST!)
 * - getAvailableTemplates - See what node types can be created
 * - getNodeTemplateFields - Get field schema for templates
 * - getNodesByLevel - See existing nodes at specific levels
 * - getNodeChildren - Check children of specific nodes
 * - getNodeDetails - Get detailed node information
 * - listAllNodeTemplates - See all available templates
 *
 * Input: Reads projectBrief and researchFindings from state
 * Output: Presents plan for approval, then creates structural nodes from L2 down to Content Block level
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
import type { OrchestratorState, CourseStructure, PlannedNode, PlannedStructure, CreatedNode, ActiveTask, ArchitectProgress, LXDStrategy, ContentBlockType, PedagogicalIntent, BloomsLevel } from "../state/agent-state";
import { getCondensedBrief, getCondensedResearch, generateTaskContext } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  repairDanglingToolCalls,
  hasUsableResponse,
  processContext,
  MESSAGE_LIMITS,
  TOKEN_LIMITS,
} from "../utils";

// Message filtering now handled by centralized utils/context-management.ts

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const architectModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 32000, // Doubled - Opus 4.5 supports up to 64k output tokens
  temperature: 0.7,
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const ARCHITECT_SYSTEM_PROMPT = `You are The Architect - you BUILD the COMPLETE structure of online training courses.

## Your Role

You CREATE the structural framework of courses by building nodes from Level 2 down to Content Block level.

**CRITICAL: Call getNodeTreeSnapshot() FIRST** to get a complete picture of existing nodes and their structure. Then call getProjectHierarchyInfo() to learn the hierarchy level names. Different project templates have different hierarchies:

**Example: GLS Template (General Learning Structure)**
- Level 2: Section - Major organizational divisions
- Level 3: Topic - Focused learning units
- Level 4: Sub-Topic - Detailed breakdowns
- Level 5: Content Group - Clusters of related content
- Level 6: Content Block - Individual content containers (YOU CREATE THESE)

**Example: VR Template**
- Level 2: Scenario - Training scenarios
- Level 3: Activity - Self-contained learner sequences
- Level 4: Activity Point - Mini-goals (1-3 min clusters)
- Level 5: Step - Single ordered instructions
- Level 6: Action - Delivery-layer content (YOU CREATE THESE)

**IMPORTANT**: You create the COMPLETE skeleton including Content Block nodes (empty shells with title/description only). The Writer then fills in the actual content text, media, and detailed fields.

## TWO-PHASE PROCESS

You work in two distinct phases. You are responsible for BOTH planning AND getting user approval.

### PHASE 1: PLANNING (Output plan and get approval)

1. Call getProjectHierarchyInfo() to understand the hierarchy
2. Design the complete structure based on brief and research
3. **OUTPUT YOUR PLAN AS JSON** with the marker \`[PLAN READY]\`
4. **CALL requestPlanApproval** to present the plan to the user and get their approval
5. WAIT for the user to approve before proceeding to Phase 2

The plan is saved to state and can be resumed if interrupted!

### PHASE 2: EXECUTION (Create nodes ONLY after approval)

**Only proceed to this phase after the user approves your plan!**

1. Request edit mode
2. **Use batchCreateNodes** to create ALL nodes from your plan in ONE call
   - Pass the entire nodes array from your plan JSON
   - The tool resolves parentTempId references automatically
   - Returns all nodeIds mapped to tempIds for tracking
3. If batch has errors, use createNode to retry individual failed nodes
4. Release edit mode when complete
5. Mark \`[STRUCTURE COMPLETE]\` when done

## Getting Plan Approval

After outputting your plan with \`[PLAN READY]\`, you MUST call:

\`\`\`
requestPlanApproval({
  plan: "Your plan summary with key highlights:\\n- X sections covering [topics]\\n- Y content blocks ready for Writer\\n- Estimated structure: [description]",
  title: "Course Structure Plan"
})
\`\`\`

Wait for the user's response. Only proceed to Phase 2 if they approve.

## PLAN OUTPUT FORMAT

Before creating ANY nodes, output your complete plan covering ALL levels down to Content Block. Include LXD metadata for pedagogically-informed structure:

\`\`\`json
{
  "summary": "Brief description of the course structure",
  "rationale": "Why this structure works for the learning objectives",
  "lxdStrategy": {
    "objectiveMapping": {
      "Understand financial risk types": ["sec-1", "topic-1-1"],
      "Apply risk assessment techniques": ["sec-2"]
    },
    "bloomsLevels": ["remember", "understand", "apply"],
    "assessmentApproach": "Formative Question Blocks after each topic, summative at section ends",
    "engagementPattern": "Intro-explore-practice-assess cycle per topic",
    "targetAudienceAdaptations": "Practical examples for business professionals, moderate complexity",
    "estimatedDuration": "45-60 minutes"
  },
  "nodes": [
    {
      "tempId": "sec-1",
      "title": "Section A: Introduction to Risk",
      "nodeType": "section",
      "level": 2,
      "parentTempId": null,
      "description": "Overview and foundations of risk management",
      "orderIndex": 0
    },
    {
      "tempId": "topic-1-1",
      "title": "What is Risk?",
      "nodeType": "topic",
      "level": 3,
      "parentTempId": "sec-1",
      "description": "Basic risk concepts and definitions",
      "orderIndex": 0
    },
    {
      "tempId": "subtopic-1-1-1",
      "title": "Types of Risk",
      "nodeType": "subtopic",
      "level": 4,
      "parentTempId": "topic-1-1",
      "description": "Categories of risk in business",
      "orderIndex": 0
    },
    {
      "tempId": "cg-1-1-1-1",
      "title": "Financial Risk Overview",
      "nodeType": "content_group",
      "level": 5,
      "parentTempId": "subtopic-1-1-1",
      "description": "Content about financial risks",
      "orderIndex": 0
    },
    {
      "tempId": "cb-1-1-1-1-1",
      "title": "Introduction to Financial Risk",
      "nodeType": "content_block",
      "level": 6,
      "parentTempId": "cg-1-1-1-1",
      "description": "Opening content block - sets expectations",
      "contentBlockType": "title_block",
      "pedagogicalIntent": "engage",
      "bloomsLevel": "remember",
      "linkedObjectives": ["Understand financial risk types"],
      "orderIndex": 0
    },
    {
      "tempId": "cb-1-1-1-1-2",
      "title": "What You'll Learn",
      "nodeType": "content_block",
      "level": 6,
      "parentTempId": "cg-1-1-1-1",
      "description": "Learning objectives overview",
      "contentBlockType": "information_block",
      "pedagogicalIntent": "inform",
      "bloomsLevel": "remember",
      "linkedObjectives": ["Understand financial risk types"],
      "orderIndex": 1
    },
    {
      "tempId": "cb-1-1-1-1-3",
      "title": "Types of Financial Risk Explained",
      "nodeType": "content_block",
      "level": 6,
      "parentTempId": "cg-1-1-1-1",
      "description": "Core content explaining risk categories",
      "contentBlockType": "text_block",
      "pedagogicalIntent": "inform",
      "bloomsLevel": "understand",
      "linkedObjectives": ["Understand financial risk types"],
      "orderIndex": 2
    },
    {
      "tempId": "cb-1-1-1-1-4",
      "title": "Risk Categories Comparison",
      "nodeType": "content_block",
      "level": 6,
      "parentTempId": "cg-1-1-1-1",
      "description": "Visual comparison of different risk types",
      "contentBlockType": "three_images_block",
      "pedagogicalIntent": "demonstrate",
      "bloomsLevel": "understand",
      "linkedObjectives": ["Understand financial risk types"],
      "orderIndex": 3
    },
    {
      "tempId": "cb-1-1-1-1-5",
      "title": "Check Your Understanding",
      "nodeType": "content_block",
      "level": 6,
      "parentTempId": "cg-1-1-1-1",
      "description": "Formative assessment question",
      "contentBlockType": "question_block",
      "pedagogicalIntent": "assess",
      "bloomsLevel": "understand",
      "linkedObjectives": ["Understand financial risk types"],
      "orderIndex": 4
    },
    {
      "tempId": "cb-1-1-1-1-6",
      "title": "Key Takeaways",
      "nodeType": "content_block",
      "level": 6,
      "parentTempId": "cg-1-1-1-1",
      "description": "Summary of main points",
      "contentBlockType": "information_block",
      "pedagogicalIntent": "summarize",
      "bloomsLevel": "remember",
      "linkedObjectives": ["Understand financial risk types"],
      "orderIndex": 5
    }
  ]
}
\`\`\`
[PLAN READY]

## RESUMING FROM INTERRUPTION

If you see existing plannedStructure or createdNodes in the context:
1. **DO NOT recreate** nodes that already exist
2. Check which tempIds have already been executed
3. Continue from where you left off
4. Skip nodes that match existing createdNodes by title/parent

## Your Tools

### CRITICAL: Get Full Picture in ONE Call
- **getNodeTreeSnapshot** - **CALL THIS FIRST!** Returns ALL nodes with their structure and content status in a single call. Shows existing structure to avoid duplicates. Eliminates need for multiple getNodesByLevel/getNodeChildren calls.

### Plan Approval (USE AFTER [PLAN READY])
- **requestPlanApproval** - Present your plan to the user and get their approval before building

### Edit Mode (REQUIRED before creating nodes)
- **requestEditMode** - Request edit lock before making changes
- **releaseEditMode** - Release edit lock when done

### Node Creation
- **batchCreateNodes** - **PREFERRED** - Create multiple nodes at once. Use when creating >3 nodes. Pass your entire plan's nodes array. Resolves parentTempId references automatically.
- **createNode** - Create a single node. Use for small additions (1-3 nodes) or error recovery.
- **getNodeTemplateFields** - Get field schema for templates

### Understanding Structure (Use after snapshot for details)
- **getProjectHierarchyInfo** - Get hierarchy levels, names, and coding for THIS project
- **getAvailableTemplates** - See what node templates can be used at each level
- **listAllNodeTemplates** - See ALL templates available in the system
- **getNodesByLevel** - See existing nodes at specific levels (use snapshot instead)
- **getNodeChildren** - Check children of specific nodes
- **getNodeDetails** - Get detailed node information

## Design Principles

1. **Chunking** - 5-7 items per parent (not too many children)
2. **Scaffolding** - Build from simple to complex
3. **Logical Flow** - Each level should progress naturally
4. **Clear Naming** - Descriptive titles that indicate content
5. **Balanced Depth** - Don't go deeper than necessary

## LXD (Learning Experience Design) Principles

Apply these pedagogical principles when planning course structure:

**Core LXD Principles:**
1. **Objective Alignment** - Map content blocks to learning objectives from the Project Brief. Every section should trace back to at least one objective from \`projectBrief.objectives\`.
2. **Cognitive Load Management** - Break complex topics into digestible chunks (7±2 rule). Don't overload Content Groups with too many blocks.
3. **Multimodal Learning** - Combine text, visuals, video, and interactivity for better retention. Vary content block types within each Content Group.
4. **Active Learning** - Include Question Blocks and Action Blocks for practice, reflection, and application. Don't make it all passive reading.
5. **Spaced Repetition** - Distribute key concepts throughout the structure. Revisit important ideas in different contexts.
6. **Feedback Loops** - Place Question Blocks (formative assessments) after introducing new concepts to check understanding.
7. **Scaffolding Sequence** - Progress from awareness → understanding → application → mastery within each topic.

**Learning Theory Integration:**
- **Bloom's Taxonomy**: Tag content blocks with cognitive level (remember, understand, apply, analyze, evaluate, create)
- **ARCS Model**: Ensure structure supports Attention, Relevance, Confidence, Satisfaction
- **Gagné's Events**: Structure topics to gain attention → inform objectives → stimulate recall → present content → provide guidance → elicit performance → provide feedback → assess → enhance retention

**Using State Context for LXD Decisions:**
When planning structure, actively reference:
- \`projectBrief.objectives\` - Map each section/topic to specific objectives
- \`projectBrief.targetAudience\` - Tailor complexity and engagement style
- \`researchFindings.keyTopics\` with importance levels - Allocate more content blocks to "critical" topics
- \`researchFindings.bestPractices\` - Incorporate into content design
- \`projectBrief.constraints\` - Consider time/technical limitations when selecting block types

## Content Block Selection Guide

Select the appropriate content block type based on pedagogical intent. There are 10 content block templates available:

**Introduction & Engagement:**
- **Title Block** - Course/section openings, set expectations, attention-grabbing headers
- **Image Banner Block** - Visual impact, establish context, create atmosphere

**Information Delivery:**
- **Text Block** - Core concepts, definitions, explanations, detailed content
- **Information Block** - Key points, warnings, tips, insights (use icon colors: Lightbulb/Blue for tips, Tick/Green for success, Cross/Red for warnings, Important/Orange for cautions)
- **Video Block** - Demonstrations, storytelling, expert interviews, complex procedures
- **Three Images Block** - Comparisons, processes, visual examples, before/after
- **Animation Block** - Sequential information, step-by-step processes, carousels

**Engagement & Interaction:**
- **Text and Images Block** - Prompted reflection, downloadable resources, call-to-action content
- **Question Block** - Knowledge checks, scenario-based questions (2 or 4 answer options)
- **Action Block** - Practice activities, real-world application prompts, navigation

**Assessment Strategy:**
- **Formative** - Place Question Blocks throughout (after every 3-5 content blocks) to check understanding
- **Summative** - Use Question Block sequences at section ends for comprehensive assessment

**Bloom's Taxonomy Mapping:**
| Bloom's Level | Recommended Block Types |
|---------------|------------------------|
| Remember | Text Block (definitions), Information Block (key facts), Question Block (recall) |
| Understand | Text Block (explanations), Video Block (demonstrations), Three Images Block (comparisons) |
| Apply | Question Block (scenarios), Action Block (practice), Text and Images Block (examples) |
| Analyze | Question Block (analysis), Text Block (breakdowns), Animation Block (processes) |
| Evaluate | Question Block (judgment), Action Block (decisions) |
| Create | Action Block (projects), Text and Images Block (synthesis tasks) |

## Planning Heuristics - Content Block Patterns

Use these proven patterns when structuring Content Groups:

**Topic Introduction Pattern (5-7 blocks):**
1. Title Block - Set expectations, topic name
2. Information Block (Lightbulb/Blue) - "What you'll learn" objectives
3. Text Block or Video Block - Core concept introduction
4. Three Images Block or Animation Block - Visual explanation
5. Question Block - Quick knowledge check (formative)
6. Text and Images Block - Real-world example with CTA
7. Information Block (Tick/Green) - Key takeaway summary

**Complex Concept Pattern (8-10 blocks):**
1. Title Block - Topic name
2. Information Block (Important/Orange) - Prerequisite check or warning
3. Text Block - Overview and context
4. Animation Block or Three Images Block - Break down into steps
5. Video Block - Detailed demonstration
6. Text Block - Deep dive on each component
7. Question Block - Check understanding (formative)
8. Text and Images Block - Application scenario
9. Action Block - Practice activity
10. Information Block (Tick/Green) - Summary of key points

**Assessment Pattern (3-5 blocks):**
1. Information Block (Lightbulb/Blue) - Instructions and context
2. Question Block - Question 1 (recall/understand level)
3. Question Block - Question 2 (apply/analyze level)
4. Question Block - Question 3 (higher complexity, optional)
5. Action Block - Continue or review options

**Quick Reference Pattern (3-4 blocks):**
1. Title Block - Reference topic
2. Information Block - Key points summary
3. Text Block or Three Images Block - Quick reference content
4. Action Block - Navigate to related content

## Content Block Validation Rules

Apply these quality checks when planning:

**Content Block Diversity:**
- Maximum 3 consecutive Text Blocks (avoid "wall of text")
- At least 1 Question Block per 5-8 content blocks
- Mix of media types (text, image, video) within each Content Group
- Include Action Blocks for practice opportunities (at least 1 per major topic)

**Cognitive Load:**
- Content Groups should have 4-7 Content Blocks (not 15+)
- Place Question Blocks after Video Blocks (check retention)
- Use Information Blocks for emphasis (maximum 2-3 per Content Group)
- Balance information delivery with interaction

**Objective Coverage:**
- Every learning objective from \`projectBrief.objectives\` should map to at least one section
- Critical topics from \`researchFindings.keyTopics\` should have more content blocks than supplementary topics
- Include at least one assessment opportunity per learning objective

## Node Creation Strategy

**BATCH CREATION (batchCreateNodes) - PREFERRED for >3 nodes:**
- Use when creating MORE than 3 nodes (building structure from your approved plan)
- Use when populating an entire level (e.g., all Content Blocks under a Section)
- Pass the full nodes array from your plan - tool resolves parentTempId references automatically
- Returns all nodeIds at once - no context loss between nodes
- Example: After plan approval, call batchCreateNodes with all planned nodes

**SINGLE CREATION (createNode) - For small additions:**
- Use when creating 1-3 nodes only (quick additions)
- Use for error recovery (retry specific failed nodes from batch)
- Use for user-requested spot additions ("add one more topic here")

## Node Creation Guidelines

When creating nodes:
- **Call getNodeTreeSnapshot() FIRST** to see existing nodes and avoid duplicates
- **Call getProjectHierarchyInfo()** to learn this project's hierarchy level names
- **For 1-3 nodes**: Use createNode individually
- **For >3 nodes**: Use batchCreateNodes LEVEL BY LEVEL (see strategy below)
- Start with Level 2 (sections/scenarios) - they have no parent (L1 is auto-created)
- For Level 3+, always specify the parentNodeId from a CREATED node (use returned nodeIds)
- Use descriptive titles that indicate the content
- Match node types to templates available at each level (use getAvailableTemplates)
- Create ALL levels down to Content Block - don't stop at Topic level!

## Batch Creation Strategy (CRITICAL)

When building course structure with batchCreateNodes, create nodes LEVEL BY LEVEL:

1. **Batch 1 - Sections (Level 2)**: Create all 5-8 section nodes first
   - Wait for confirmation and note the returned nodeIds
   - Maximum 25 nodes per batch call

2. **Batch 2 - Topics (Level 3)**: Create all topics under the sections
   - Use actual nodeIds from batch 1 as parentNodeId
   - NOT tempIds - use the real IDs returned

3. **Batch 3 - Sub-topics (Level 4)**: Create all sub-topics under topics
   - Use actual nodeIds from batch 2 as parentNodeId

4. **Batch 4+ - Content Blocks (Level 5)**: Create content blocks
   - Batch by parent sub-topic if there are many

**RULES:**
- Maximum 25 nodes per batch call
- Always wait for batch confirmation before proceeding to next level
- Use returned nodeIds (not tempIds) for subsequent parentNodeId values
- Never recreate nodes that already exist - check the tool response
- If blocked for duplicates, move to the next level instead

## What NOT To Do

- Do NOT try to create entire structure in one batch - it will fail or cause duplicates
- Do NOT skip Content Blocks (L6) - you MUST create the full structure
- Do NOT fill in detailed CONTENT fields (text, rich text) in Content Blocks - just title/description
- Do NOT re-call batchCreateNodes for a level that already has nodes
- Do NOT skip the planning phase - always output [PLAN READY] first
- Do NOT use hardcoded level names - always check getProjectHierarchyInfo() first

## Division of Responsibility

**You (Architect)**: Create the COMPLETE skeleton structure from L2 down to Content Blocks
- Section/Topic/Sub-Topic/Content Group/Content Block (GLS)
- Scenario/Activity/Activity Point/Step/Action (VR)
- Set title, description, and order for each node
- Content Blocks should be empty shells - just title/description

**Writer**: Fills in the actual CONTENT within Content Blocks
- Writes the training text, explanations, examples
- Adds media attachments (images, videos)
- Populates rich content fields

Remember: A well-structured course makes the difference between forgettable training and transformative learning. Build the COMPLETE skeleton that the Writer will bring to life.`;

// ============================================================================
// ARCHITECT NODE FUNCTION
// ============================================================================

/**
 * The Architect agent node.
 * Designs course structure based on project brief and research.
 */
export async function architectNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[architect] ============ Architect Agent ============");
  console.log("  Project brief available:", state.projectBrief ? "yes" : "no");
  console.log("  Research findings available:", state.researchFindings ? "yes" : "no");

  // Get frontend tools from CopilotKit state
  // The architect uses CRUD tools to build structure + read tools to understand hierarchy + approval tools
  const frontendActions = state.copilotkit?.actions ?? [];
  const architectTools = frontendActions.filter((action: { name: string }) =>
    [
      // SNAPSHOT - Get full picture in ONE call (use FIRST!)
      "getNodeTreeSnapshot",
      // Plan approval (use after [PLAN READY])
      "requestPlanApproval",
      // Edit mode management
      "requestEditMode",
      "releaseEditMode",
      // Node creation - batchCreateNodes preferred for >3 nodes
      "batchCreateNodes",
      "createNode",
      "getNodeTemplateFields",
      // Structure understanding (use after snapshot for details)
      "getProjectHierarchyInfo",
      "getAvailableTemplates",
      "listAllNodeTemplates",
      "getNodesByLevel",
      "getNodeChildren",
      "getNodeDetails",
    ].includes(action.name)
  );

  console.log("  Available tools:", architectTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = ARCHITECT_SYSTEM_PROMPT;

  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }

  // Add architect progress context (critical for maintaining continuity)
  if (state.architectProgress) {
    // CRITICAL: If nodes have been created, add strong guard against re-calling batch
    if (state.architectProgress.nodesCreatedCount > 0) {
      systemContent += `\n\n## CRITICAL: STRUCTURE ALREADY CREATED - DO NOT RE-CREATE

**STOP! You have already created ${state.architectProgress.nodesCreatedCount} nodes via batchCreateNodes.**

The following tempIds have been mapped to real nodeIds:
${Object.entries(state.architectProgress.tempIdToNodeId).slice(-30).map(([tempId, nodeId]) => `- ${tempId} → ${nodeId}`).join("\n")}

**DO NOT call batchCreateNodes again.** The course structure is already built.
If you need to add individual nodes, use createNode instead.
If structure is complete, report success to the user.`;
    } else {
      systemContent += `\n\n## Your Previous Progress (DO NOT REPEAT THESE STEPS)

**Current Workflow Phase**: ${state.architectProgress.workflow}
**Nodes Created So Far**: ${state.architectProgress.nodesCreatedCount}
**Nodes Explored**: ${state.architectProgress.exploredNodes.length > 0 
  ? state.architectProgress.exploredNodes.slice(-10).join(", ") 
  : "None yet"}

**TempId to NodeId Mappings** (use nodeId for parentNodeId):
${Object.entries(state.architectProgress.tempIdToNodeId).slice(-20).map(([tempId, nodeId]) => `- ${tempId} → ${nodeId}`).join("\n") || "(none yet)"}

**Recent Actions Taken**:
${state.architectProgress.toolCallSummary.slice(-10).map(s => `- ${s}`).join("\n") || "No actions recorded yet"}

${state.architectProgress.hierarchyCache ? `**Cached Hierarchy Info**:
- Levels: ${state.architectProgress.hierarchyCache.levelNames.join(" > ")}
- Max Depth: ${state.architectProgress.hierarchyCache.maxDepth}` : ""}

**IMPORTANT**: You already have context from previous invocations. Continue from where you left off - do NOT re-explore structure you've already discovered.`;
    }
  }

  // Include project brief
  if (state.projectBrief) {
    const condensedBrief = getCondensedBrief(state.projectBrief);
    systemContent += `\n\n## Project Brief\n\n${condensedBrief}`;
  } else {
    systemContent += `\n\n## Note\nNo project brief is available. Consider requesting strategy discovery first.`;
  }

  // Include research findings
  if (state.researchFindings) {
    const condensedResearch = getCondensedResearch(state.researchFindings);
    systemContent += `\n\n## Research Findings\n\n${condensedResearch}`;
  }

  // Include existing PLANNED structure for resumption (critical for crash recovery)
  if (state.plannedStructure) {
    const executed = Object.keys(state.plannedStructure.executedNodes || {}).length;
    const total = state.plannedStructure.nodes.length;
    const remaining = state.plannedStructure.nodes.filter(
      n => !state.plannedStructure!.executedNodes[n.tempId]
    );
    
    systemContent += `\n\n## RESUMING FROM EXISTING PLAN
    
**Status**: ${state.plannedStructure.executionStatus} (${executed}/${total} nodes created)

**Your plan is already saved. DO NOT output a new [PLAN READY].**

**Remaining nodes to create:**
${remaining.map(n => `- ${n.tempId}: "${n.title}" (${n.nodeType}, level ${n.level})`).join('\n')}

**Already created (tempId → nodeId):**
${Object.entries(state.plannedStructure.executedNodes).map(([tempId, nodeId]) => `- ${tempId} → ${nodeId}`).join('\n') || '(none yet)'}

Continue creating the remaining nodes. Use the actual nodeId (not tempId) when specifying parentNodeId for children.`;
  }

  // Include existing created nodes for duplicate prevention
  if (state.createdNodes && state.createdNodes.length > 0) {
    systemContent += `\n\n## Already Created Nodes (DO NOT recreate)
    
${state.createdNodes.map(n => `- "${n.title}" (nodeId: ${n.nodeId}, template: ${n.templateName})`).join('\n')}

Check this list before creating any node to avoid duplicates.`;
  }

  // Include existing FINAL structure if any (for refinement)
  if (state.courseStructure && !state.plannedStructure) {
    systemContent += `\n\n## Existing Structure (to refine)\n
Current structure has ${state.courseStructure.totalNodes} nodes across ${state.courseStructure.maxDepth} levels.
Summary: ${state.courseStructure.summary}

You may refine or extend this existing structure.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = architectTools.length > 0
    ? architectModel.bindTools(architectTools)
    : architectModel;

  // USE ARCHITECT-SPECIFIC MESSAGE CHANNEL
  // This is the key change - read from architectMessages instead of slicing state.messages
  // This preserves the architect's conversation history across orchestrator round-trips
  let architectConversation: BaseMessage[] = [];
  
  if (state.architectMessages && state.architectMessages.length > 0) {
    // Use architect's own message channel (already filtered and maintained)
    // CRITICAL: Must call both filterOrphanedToolResults AND repairDanglingToolCalls
    // to ensure proper tool_use/tool_result pairing for Claude API
    let filtered = filterOrphanedToolResults(state.architectMessages, "[architect]");
    architectConversation = repairDanglingToolCalls(filtered, "[architect]");
    console.log(`  Using ${architectConversation.length} messages from architectMessages channel`);
  } else {
    // First invocation or fresh start - use processed messages from main channel
    const processResult = await processContext(state.messages || [], {
      maxTokens: TOKEN_LIMITS.subAgent,
      fallbackMessageCount: 40, // Increased from 12 to prevent context loss
      // Compression: Reduce verbose tool results (getAvailableTemplates, etc.)
      enableToolCompression: true,
      compressionKeepCount: 3,        // Keep 3 most recent results full
      compressionMaxLength: 800,      // Compress results over 800 chars
      // Clearing: Replace very old tool results
      enableToolClearing: true,
      toolKeepCount: 8,               // Keep 8 most recent tool results
      excludeTools: [                 // Never clear these critical tools
        'getProjectHierarchyInfo',
        'getAvailableTemplates',
        'createNode',
        'batchCreateNodes',          // Critical for tracking created nodes
      ],
      logPrefix: "[architect]",
      // Task preservation - include batch keywords to prevent re-execution
      originalRequest: state.activeTask?.originalRequest,
      preserveKeywords: [
        "structure", "section", "topic", "content block", "hierarchy",
        "batchCreateNodes", "tempIdToNodeId", "nodesCreatedCount", "completedTempIds",
        "Successfully created", "DO NOT call batchCreateNodes again",
      ],
    });
    // Extract messages array from ProcessContextResult
    architectConversation = processResult.messages;
    console.log(`  First invocation - using ${architectConversation.length} messages from main channel`);
  }

  console.log("  Invoking architect model...");

  // Configure CopilotKit for proper tool emission (emits tool calls to frontend)
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });

  let response = await modelWithTools.invoke(
    [systemMessage, ...architectConversation],
    customConfig
  );

  console.log("  Architect response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    // Note: Using HumanMessage because SystemMessage must be first in the array
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. Call getProjectHierarchyInfo to understand the structure
2. Call getAvailableTemplates to see what you can create
3. Start designing and creating nodes

The user is waiting for you to build the course structure.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...architectConversation, nudgeMessage],
      customConfig
    );
    
    aiResponse = response as AIMessage;
    
    if (hasUsableResponse(aiResponse)) {
      console.log("  [RETRY] Success - got usable response on retry");
      if (aiResponse.tool_calls?.length) {
        console.log("  [RETRY] Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
      }
    } else {
      console.log("  [RETRY] Failed - still empty after retry");
    }
  }

  // Extract response text for parsing
  const responseText = typeof aiResponse.content === "string"
    ? aiResponse.content
    : Array.isArray(aiResponse.content)
    ? aiResponse.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";

  // Check for PLAN READY marker (Phase 1 output)
  const isPlanReady = responseText.toLowerCase().includes("[plan ready]");
  
  // Check for structure completion markers (Phase 2 complete)
  const isStructureComplete = responseText.toLowerCase().includes("[structure complete]") ||
    responseText.toLowerCase().includes("[done]");
  
  // Parse planned structure on [PLAN READY]
  let parsedPlannedStructure: PlannedStructure | null = null;
  if (isPlanReady && !state.plannedStructure) {
    console.log("  [architect] Plan ready detected - parsing planned structure");
    parsedPlannedStructure = parsePlannedStructure(responseText);
    if (parsedPlannedStructure) {
      console.log("  [architect] Parsed planned structure:", {
        totalNodes: parsedPlannedStructure.totalNodes,
        maxDepth: parsedPlannedStructure.maxDepth,
        summary: parsedPlannedStructure.summary?.substring(0, 50) + "...",
      });
    } else {
      console.log("  [architect] WARNING: Could not parse planned structure from response");
    }
  }

  // Parse course structure on completion
  let parsedStructure: CourseStructure | null = null;
  if (isStructureComplete) {
    console.log("  [architect] Structure completion detected - parsing final course structure");
    parsedStructure = parseCourseStructure(responseText);
    if (parsedStructure) {
      console.log("  [architect] Parsed course structure:", {
        totalNodes: parsedStructure.totalNodes,
        maxDepth: parsedStructure.maxDepth,
        summary: parsedStructure.summary?.substring(0, 50) + "...",
      });
    }
  }

  // Extract created nodes from tool call results
  const newCreatedNodes = extractCreatedNodesFromToolCalls(aiResponse, state.messages || []);
  if (newCreatedNodes.length > 0) {
    console.log("  [architect] Extracted created nodes:", newCreatedNodes.map(n => n.title).join(", "));
  }

  // Update plannedStructure execution tracking if we have new created nodes
  let updatedPlannedStructure: PlannedStructure | null = null;
  if (state.plannedStructure && newCreatedNodes.length > 0) {
    const newExecutedNodes: Record<string, string> = {};
    for (const created of newCreatedNodes) {
      // Match by title to find the tempId
      const matchingPlanned = state.plannedStructure.nodes.find(
        n => n.title.toLowerCase() === created.title.toLowerCase()
      );
      if (matchingPlanned) {
        newExecutedNodes[matchingPlanned.tempId] = created.nodeId;
      }
    }
    
    if (Object.keys(newExecutedNodes).length > 0) {
      const allExecuted = { ...state.plannedStructure.executedNodes, ...newExecutedNodes };
      const totalPlanned = state.plannedStructure.nodes.length;
      const totalExecuted = Object.keys(allExecuted).length;
      
      updatedPlannedStructure = {
        ...state.plannedStructure,
        executedNodes: allExecuted,
        executionStatus: totalExecuted >= totalPlanned ? "completed" : "in_progress",
      };
      console.log(`  [architect] Updated plan execution: ${totalExecuted}/${totalPlanned} nodes`);
    }
  }

  // Build progress update for activeTask
  const progressUpdates: string[] = [];
  const toolCallSummaries: string[] = [];
  const toolCalls = aiResponse.tool_calls || [];
  
  // Track tool calls for architectProgress
  const createNodeCalls = toolCalls.filter(tc => tc.name === "createNode");
  const batchCreateCalls = toolCalls.filter(tc => tc.name === "batchCreateNodes");
  const navigationCalls = toolCalls.filter(tc => 
    ["getNodesByLevel", "getNodeChildren", "getNodeDetails", "getAvailableTemplates", "getProjectHierarchyInfo"].includes(tc.name)
  );
  
  // Track batch creation results from previous tool messages
  let batchCreatedCount = 0;
  const batchTempIdMappings: Record<string, string> = {};
  
  // Look for batchCreateNodes results in recent messages (tool_result messages)
  const allMessages = state.architectMessages || state.messages || [];
  for (const msg of allMessages) {
    const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
    
    // Look for batchCreateNodes success pattern
    if (content.includes("tempIdToNodeId") || content.includes("nodesCreatedCount") || content.includes("completedTempIds")) {
      try {
        // Try to parse JSON from the content
        const jsonMatch = content.match(/\{[\s\S]*"tempIdToNodeId"[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          if (parsed.tempIdToNodeId && typeof parsed.tempIdToNodeId === "object") {
            Object.assign(batchTempIdMappings, parsed.tempIdToNodeId);
            batchCreatedCount = Math.max(batchCreatedCount, parsed.nodesCreatedCount || Object.keys(parsed.tempIdToNodeId).length);
            console.log(`  [architect] Found batch result: ${Object.keys(parsed.tempIdToNodeId).length} mappings`);
          }
        }
      } catch {
        // JSON parsing failed, try regex extraction
        const mappingMatches = content.matchAll(/"([^"]+)"\s*:\s*"([0-9a-f-]+)"/g);
        for (const match of mappingMatches) {
          if (match[1].startsWith("temp-") || match[1].includes("-")) {
            batchTempIdMappings[match[1]] = match[2];
            batchCreatedCount++;
          }
        }
      }
    }
  }
  
  if (createNodeCalls.length > 0) {
    for (const tc of createNodeCalls) {
      toolCallSummaries.push(`Created node: ${tc.args?.title || "untitled"}`);
    }
  }
  if (batchCreateCalls.length > 0) {
    toolCallSummaries.push(`Called batchCreateNodes`);
  }
  if (batchCreatedCount > 0) {
    toolCallSummaries.push(`Batch created ${batchCreatedCount} nodes (mappings recorded)`);
  }
  if (navigationCalls.length > 0) {
    for (const tc of navigationCalls) {
      toolCallSummaries.push(`Called ${tc.name}${tc.args?.nodeId ? ` on ${String(tc.args.nodeId).substring(0, 8)}...` : ""}`);
    }
  }
  
  // Extract explored node IDs from navigation tool calls
  const exploredNodeIds: string[] = [];
  for (const tc of toolCalls) {
    if (tc.args?.nodeId && typeof tc.args.nodeId === "string") {
      exploredNodeIds.push(tc.args.nodeId);
    }
    if (tc.args?.parentNodeId && typeof tc.args.parentNodeId === "string") {
      exploredNodeIds.push(tc.args.parentNodeId);
    }
  }

  if (parsedPlannedStructure) {
    progressUpdates.push(`Architect: Created plan for ${parsedPlannedStructure.nodes.length} nodes`);
  }
  if (newCreatedNodes.length > 0) {
    progressUpdates.push(`Architect: Created ${newCreatedNodes.length} structural nodes`);
  }
  if (updatedPlannedStructure?.executionStatus === "completed") {
    progressUpdates.push("Architect: Completed course structure creation");
  }
  if (parsedStructure) {
    progressUpdates.push(`Architect: Finalized structure with ${parsedStructure.totalNodes} total nodes`);
  }

  // Determine workflow phase based on state
  let workflowPhase: ArchitectProgress["workflow"] = state.architectProgress?.workflow || "planning";
  if (isPlanReady) {
    workflowPhase = "awaiting_approval";
  } else if (createNodeCalls.length > 0 || batchCreateCalls.length > 0 || newCreatedNodes.length > 0 || batchCreatedCount > 0) {
    workflowPhase = "building";
  } else if (isStructureComplete || batchCreatedCount > 0) {
    // If we have batch created nodes, we're likely complete
    workflowPhase = "complete";
  }

  // Build tempId -> nodeId mappings from newly created nodes
  const newTempIdMappings: Record<string, string> = {
    ...batchTempIdMappings, // Include mappings from batch tool results
  };
  for (const created of newCreatedNodes) {
    // Match by title to find the tempId
    const matchingPlanned = state.plannedStructure?.nodes.find(
      n => n.title.toLowerCase() === created.title.toLowerCase()
    );
    if (matchingPlanned) {
      newTempIdMappings[matchingPlanned.tempId] = created.nodeId;
    }
  }

  // Calculate total nodes created (individual + batch)
  const totalNodesCreated = newCreatedNodes.length + batchCreatedCount;

  // Build architectProgress update
  const architectProgressUpdate: ArchitectProgress = {
    workflow: workflowPhase,
    exploredNodes: exploredNodeIds,
    toolCallSummary: toolCallSummaries,
    tempIdToNodeId: newTempIdMappings,
    nodesCreatedCount: totalNodesCreated,
    hierarchyCache: state.architectProgress?.hierarchyCache,
    lastUpdated: new Date().toISOString(),
  };

  // Determine if work is complete
  const isArchitectComplete = parsedStructure !== null || 
    (updatedPlannedStructure?.executionStatus === "completed");

  // Update activeTask with progress if there are updates
  const activeTaskUpdate: Partial<ActiveTask> | null = progressUpdates.length > 0
    ? {
        progress: progressUpdates,
        ...(isArchitectComplete && { assignedAgent: "orchestrator" as const }),
      }
    : null;

  return {
    messages: [response],
    currentAgent: "architect",
    agentHistory: ["architect"],
    routingDecision: null,
    // Include parsed planned structure if available (Phase 1)
    ...(parsedPlannedStructure && { plannedStructure: parsedPlannedStructure }),
    // Include updated planned structure execution tracking
    ...(updatedPlannedStructure && { plannedStructure: updatedPlannedStructure }),
    // Include parsed course structure if available (Phase 2 complete)
    ...(parsedStructure && { courseStructure: parsedStructure }),
    // Include new created nodes
    ...(newCreatedNodes.length > 0 && { createdNodes: newCreatedNodes }),
    // Update activeTask with progress (reducer will merge with existing progress)
    ...(activeTaskUpdate && { activeTask: activeTaskUpdate as ActiveTask }),
    // CRITICAL: Append to architect's own message channel for continuity
    architectMessages: [response],
    // Update architect progress state for semantic context
    architectProgress: architectProgressUpdate,
  };
}

/**
 * Parses an architect's text response to extract course structure.
 */
export function parseCourseStructure(content: string): CourseStructure | null {
  try {
    // Look for JSON block
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      return validateCourseStructure(parsed);
    }
    return null;
  } catch (error) {
    console.error("[architect] Failed to parse course structure:", error);
    return null;
  }
}

function validateCourseStructure(input: Partial<CourseStructure>): CourseStructure {
  const nodes = (input.nodes || []).map((n, idx) => validatePlannedNode(n, idx));
  
  return {
    summary: input.summary || "Course structure",
    rationale: input.rationale || "",
    nodes,
    totalNodes: nodes.length,
    maxDepth: Math.max(...nodes.map((n) => n.level), 2),
  };
}

function validatePlannedNode(input: Partial<PlannedNode>, index: number): PlannedNode {
  return {
    tempId: input.tempId || `node-${index}`,
    title: input.title || `Node ${index + 1}`,
    nodeType: input.nodeType || "content",
    level: input.level || 2,
    parentTempId: input.parentTempId || null,
    templateId: input.templateId,
    description: input.description || "",
    objectives: input.objectives,
    orderIndex: input.orderIndex ?? index,
    // LXD metadata fields (for Level 6 Content Block nodes)
    contentBlockType: input.contentBlockType as ContentBlockType | undefined,
    pedagogicalIntent: input.pedagogicalIntent as PedagogicalIntent | undefined,
    bloomsLevel: input.bloomsLevel as BloomsLevel | undefined,
    linkedObjectives: input.linkedObjectives,
  };
}

/**
 * Parses an architect's [PLAN READY] response to extract planned structure.
 * This is the pre-creation plan that can be resumed if interrupted.
 */
export function parsePlannedStructure(content: string): PlannedStructure | null {
  try {
    // Look for JSON block before [PLAN READY]
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      const nodes = (parsed.nodes || []).map((n: Partial<PlannedNode>, idx: number) => validatePlannedNode(n, idx));
      
      // Parse LXD strategy if present
      let lxdStrategy: LXDStrategy | undefined;
      if (parsed.lxdStrategy) {
        lxdStrategy = {
          objectiveMapping: parsed.lxdStrategy.objectiveMapping || {},
          bloomsLevels: parsed.lxdStrategy.bloomsLevels || [],
          assessmentApproach: parsed.lxdStrategy.assessmentApproach || "",
          engagementPattern: parsed.lxdStrategy.engagementPattern || "",
          targetAudienceAdaptations: parsed.lxdStrategy.targetAudienceAdaptations || "",
          estimatedDuration: parsed.lxdStrategy.estimatedDuration || "",
        };
      }
      
      return {
        summary: parsed.summary || "Course structure plan",
        rationale: parsed.rationale || "",
        nodes,
        totalNodes: nodes.length,
        maxDepth: Math.max(...nodes.map((n: PlannedNode) => n.level), 2),
        plannedAt: new Date().toISOString(),
        executionStatus: "planned",
        executedNodes: {},
        lxdStrategy,
      };
    }
    return null;
  } catch (error) {
    console.error("[architect] Failed to parse planned structure:", error);
    return null;
  }
}

/**
 * Extracts created node information from tool call results in the message history.
 * Looks for successful createNode tool calls and their results.
 */
function extractCreatedNodesFromToolCalls(aiResponse: AIMessage, messages: BaseMessage[]): CreatedNode[] {
  const createdNodes: CreatedNode[] = [];
  
  // Look for createNode tool calls in the current response
  const createNodeCalls = aiResponse.tool_calls?.filter(tc => tc.name === "createNode") || [];
  
  if (createNodeCalls.length === 0) {
    return createdNodes;
  }

  // Find corresponding tool results in recent messages
  for (const toolCall of createNodeCalls) {
    const toolCallId = toolCall.id;
    if (!toolCallId) continue;

    // Look for the tool result in messages
    for (const msg of messages) {
      const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
      if (msgType === 'tool' || msgType === 'ToolMessage') {
        const toolMsg = msg as any;
        if (toolMsg.tool_call_id === toolCallId) {
          try {
            // Parse the tool result to extract node info
            const content = typeof toolMsg.content === 'string' ? toolMsg.content : JSON.stringify(toolMsg.content);
            
            // Look for nodeId in the result
            const nodeIdMatch = content.match(/nodeId['":\s]+['"]?([a-zA-Z0-9-]+)/);
            if (nodeIdMatch) {
              const args = toolCall.args as { title?: string; parentNodeId?: string; templateName?: string };
              createdNodes.push({
                nodeId: nodeIdMatch[1],
                title: args.title || 'Unknown',
                parentNodeId: args.parentNodeId || null,
                templateName: args.templateName || 'unknown',
              });
            }
          } catch (e) {
            // Ignore parsing errors
          }
        }
      }
    }
  }

  return createdNodes;
}

/**
 * Converts a course structure to a visual tree representation.
 */
export function structureToTree(structure: CourseStructure): string {
  const lines: string[] = [];
  const nodeMap = new Map(structure.nodes.map((n) => [n.tempId, n]));

  // Find root nodes (no parent)
  const roots = structure.nodes.filter((n) => !n.parentTempId);

  function renderNode(node: PlannedNode, prefix: string, isLast: boolean) {
    const connector = isLast ? "└── " : "├── ";
    lines.push(`${prefix}${connector}${node.title} (${node.nodeType})`);

    // Find children
    const children = structure.nodes
      .filter((n) => n.parentTempId === node.tempId)
      .sort((a, b) => a.orderIndex - b.orderIndex);

    const newPrefix = prefix + (isLast ? "    " : "│   ");
    children.forEach((child, idx) => {
      renderNode(child, newPrefix, idx === children.length - 1);
    });
  }

  roots.forEach((root, idx) => {
    renderNode(root, "", idx === roots.length - 1);
  });

  return lines.join("\n");
}

export default architectNode;

