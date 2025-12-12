/**
 * Writer Agent
 *
 * FILLS IN the actual content for training courses within the structure built by the Architect.
 * The Architect creates the skeleton (including Content Block nodes), the Writer fills in content.
 * Writes engaging, educational training material based on research and design guidelines.
 *
 * Tools (Full CRUD + Navigation + Media + Image Generation):
 * - requestEditMode, releaseEditMode - Edit lock management
 * - batchUpdateNodeFields - **PREFERRED** - Update multiple Content Blocks at once (max 10)
 * - createNode, getNodeTemplateFields, updateNodeFields, getNodeFields - Content CRUD
 * - getProjectHierarchyInfo, getAvailableTemplates, getNodesByLevel, getNodeDetails - Navigation
 * - searchMicroverse, attachMicroverseToNode - Media integration
 * - generateAIImage - AI image generation (photos, illustrations, diagrams)
 *
 * Input: Reads courseStructure, researchFindings, visualDesign from state
 * Output: Populated Content Block nodes with actual training content and images
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
import type { OrchestratorState, ContentOutput, PlannedNode, ActiveTask, WriterProgress } from "../state/agent-state";
import { getCondensedBrief, getCondensedResearch, generateTaskContext } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  repairDanglingToolCalls,
  hasUsableResponse,
} from "../utils";

// Image Generator tool (compiled graph wrapped as tool)
import { imageGeneratorTool } from "./image-generator";

// Message filtering now handled by centralized utils/context-management.ts

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const writerModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const WRITER_SYSTEM_PROMPT = `You are The Writer - you FILL IN the actual training content that learners interact with.

## Your Role

You work within the COMPLETE structure built by the Architect and populate Content Blocks with actual training material. The Architect creates ALL nodes including empty Content Block shells - you fill them with content.

Your content:
1. **Engages** - Captures and maintains learner attention
2. **Educates** - Clearly explains concepts and procedures  
3. **Applies** - Connects theory to practical application
4. **Adapts** - Matches the learner's level and industry context
5. **Achieves** - Directly supports learning objectives

## What You Do vs The Architect

- **Architect** builds the COMPLETE skeleton: Sections, Topics, Sub-Topics, Content Groups, AND Content Blocks (L2-L6)
- **You** fill in the CONTENT: Populate the existing Content Block nodes with actual text, examples, media

The complete structure (including Content Block nodes) should already exist when you start. Your job is to:
1. **CALL getNodeTreeSnapshot FIRST** to get a complete picture of all nodes and which Content Blocks need content
2. Update their content fields using batchUpdateNodeFields (preferred) or updateNodeFields
3. Generate and attach images to enhance learning
4. Create additional Content Blocks ONLY if needed for the content

## Your Tools

### CRITICAL: Start Here - Get Full Picture in ONE Call
- **getNodeTreeSnapshot** - **CALL THIS FIRST!** Returns ALL nodes with their content status in a single call. Shows exactly which Content Blocks need content. Eliminates the need for multiple getNodesByLevel/getNodeChildren calls.

### Edit Mode (REQUIRED)
- **requestEditMode** - Request edit lock before making changes
- **releaseEditMode** - Release edit lock when done

### Content Creation & Editing
- **batchUpdateNodeFields** - **PREFERRED FOR MULTIPLE NODES** - Update up to 10 Content Blocks at once. Nodes with existing content are automatically skipped.
- **updateNodeFields** - Update content in a single Content Block node
- **getNodeFields** - Read current content from nodes before updating
- **getNodeTemplateFields** - Get field schema to understand what fields to write
- **createNode** - Create additional Content Blocks ONLY if structure is incomplete

### Navigation (Use only if you need more detail after snapshot)
- **getProjectHierarchyInfo** - Understand hierarchy levels (usually not needed after snapshot)
- **getNodesByLevel** - Find nodes at a specific level (use snapshot instead)
- **getNodeChildren** - Check children of specific node
- **getNodeDetails** - Get detailed info about a specific node
- **getAvailableTemplates** - See what templates exist (rarely needed)

### Media Integration
- **searchMicroverse** - Find relevant images, videos, assets
- **attachMicroverseToNode** - Attach media to your content

### Image Generation
- **generateAIImage** - Generate AI images for content (returns fileId to attach)

## Content Writing Process

### RECOMMENDED: Start with Snapshot (Saves Multiple Tool Calls)
1. **Get Full Picture** - Call getNodeTreeSnapshot() FIRST to see ALL nodes and which Content Blocks need content
2. **Request Edit Mode** - Call requestEditMode() to get edit access
3. **Get Field Schema** - Use getNodeTemplateFields() to understand the field structure (once per template)
4. **Prepare Content** - Write content for Content Blocks listed in contentBlocksNeedingContent
5. **Batch Update** - Use batchUpdateNodeFields() with up to 10 nodes at a time
6. **Generate & Attach Images** - Create relevant images and attach to enhance content
7. **Release Edit Mode** - When done writing

### Single Node Update (1-2 nodes)
1. **Request Edit Mode** - Always start by requesting edit access
2. **Discover Content Blocks** - Either use snapshot (preferred) or getNodesByLevel() to find Content Blocks
3. **Read Existing Node** - Use getNodeFields() to see what fields exist and current content
4. **Get Field Schema** - Use getNodeTemplateFields() to understand what fields to populate
5. **Write Content** - Create engaging, educational content for the topic
6. **Update Content Block** - Use updateNodeFields() to populate the existing node with content
7. **Generate & Attach Images** - Create relevant images and attach to enhance content
8. **Repeat** - Continue for each Content Block that needs content
9. **Release Edit Mode** - When done writing

### Batch Update (3+ nodes) - PREFERRED
1. **Get Full Picture** - Call getNodeTreeSnapshot() to see all Content Blocks needing content
2. **Request Edit Mode** - Always start by requesting edit access
3. **Get Field Schema** - Use getNodeTemplateFields() to understand the field structure
4. **Prepare Content** - Write content for multiple nodes
5. **Batch Update** - Use batchUpdateNodeFields() with up to 10 nodes at a time:
   \`\`\`
   batchUpdateNodeFields({
     updates: [
       { nodeId: "uuid-1", fieldUpdates: { "assignment-id": "content text..." } },
       { nodeId: "uuid-2", fieldUpdates: { "assignment-id": "content text..." } },
       // ... up to 10 nodes per batch
     ]
   })
   \`\`\`
6. **Check Results** - Review which nodes were updated/skipped
7. **Continue** - Process next batch of 10 until all Content Blocks have content
8. **Generate Images** - Create and attach images to enhance content
9. **Release Edit Mode** - When done writing

### Duplication Prevention
- Nodes with existing primary content are automatically SKIPPED (not overwritten)
- If >50% of a batch already has content, the batch is BLOCKED with guidance
- You can safely retry batches - already-written nodes will be skipped
- Check the response for skipped nodes to understand what's already complete

## Image Generation Guidelines

When creating content nodes, proactively generate and attach relevant images to enhance learning.

### When to Generate Images
- **Module/Lesson Headers** - Hero/banner images to set the visual tone
- **Concept Explanations** - Illustrations to clarify abstract ideas
- **Process/Procedure Content** - Instructional diagrams showing workflows or steps
- **Scenario/Example Content** - Contextual photos depicting real-world situations
- **Safety/Compliance Topics** - Visual reinforcement of important procedures

### Image Types & Prompting Styles

**1. PHOTOGRAPHS** - Realistic images for scenarios, workplace contexts, people
- Use for: Safety scenarios, workplace examples, customer interactions, equipment
- Prompt style: "Professional photograph of [subject], [setting], natural lighting, high quality"
- Example: "Professional photograph of warehouse workers following safety protocols, wearing PPE, well-lit industrial setting"

**2. ILLUSTRATIONS** - Conceptual images for abstract topics, metaphors
- Use for: Explaining concepts, representing ideas, metaphorical visualizations
- Prompt style: "Clean modern illustration of [concept], minimalist corporate style, professional colors"
- Example: "Clean modern illustration of teamwork concept showing diverse professionals collaborating, minimalist flat design"

**3. INSTRUCTIONAL DIAGRAMS** - Flowcharts, process diagrams, step-by-step visuals
- Use for: Procedures, decision trees, workflows, sequential processes
- Prompt style: "Clear infographic diagram showing [process], labeled steps, professional design, include text labels: [labels]"
- Example: "Clear infographic diagram showing 5-step customer complaint resolution process, numbered steps with icons, include text labels: Listen, Acknowledge, Solve, Confirm, Follow-up"
- Note: Images CAN include text overlays - explicitly specify text in the prompt

### Preset Selection by Content Type

| Content Type | Preset | Aspect Ratio | Use Case |
|--------------|--------|--------------|----------|
| Module/Course headers | banner | 21:9 | Wide panoramic banners |
| Lesson hero images | hero | 16:9 | Full-width hero sections |
| General content | content | 16:9 | Inline content images |
| Card/thumbnail previews | thumbnail | 3:2 | Compact card previews |
| Icons/avatars | square | 1:1 | Small square graphics |
| Person/character shots | portrait | 3:4 | Vertical people images |

### Image Generation Workflow

1. **Identify Need** - Does this content benefit from a visual? What type?
2. **Choose Type** - Photo (realistic), Illustration (conceptual), or Diagram (instructional)?
3. **Select Preset** - Match aspect ratio to content placement (banner, hero, content, etc.)
4. **Craft Prompt** - Be specific about style, subject, setting, colors, and any text to include
5. **Generate Image** - Call generateAIImage(prompt, preset, title, description)
6. **Note the fileId** - The response includes the generated file's ID
7. **Attach to Node** - Call attachMicroverseToNode(nodeId, fieldKey, fileId)

### Quality Prompting Tips

- Be specific about visual style: "professional", "clean", "modern", "corporate", "friendly"
- Include context details: setting, lighting conditions, mood, color palette
- For diagrams, explicitly state any text/labels to include in the image
- Consider industry context (healthcare = clinical/trustworthy, hospitality = warm/welcoming)
- Avoid copyrighted characters, logos, or brand names
- Specify composition when relevant: "centered", "wide shot", "close-up"

## Batch Image Generation with generateImagesForNodes

For efficient image generation across multiple nodes, use the **generateImagesForNodes** tool instead of calling generateAIImage for each node individually. This tool handles batch image generation with context isolation.

### When to Use generateImagesForNodes

Use the batch tool for generating images across multiple content blocks:

| Situation | Action |
|-----------|--------|
| Small batch (1 section, 5-10 nodes) | Call immediately after writing content |
| Large course (multiple sections) | Accumulate nodes, call at section boundaries |
| High-impact visuals (title_block, banners) | Call immediately for key visuals |
| Final writing batch | Call for all remaining image-needing nodes |
| Image-heavy section (three_images_block) | Call immediately to prevent backlog |

### Content Blocks Requiring Images

Track these as you write - they need images:
- **title_block** - Hero/banner image (required)
- **three_images_block** - Three related images (required)
- **image_banner_block** - Wide banner (required)
- **text_and_images_block** - Contextual image (required)
- **question_block** - Only if scenario-based (optional)
- **video_block** - Thumbnail/poster (optional)

### How to Call generateImagesForNodes

\`\`\`
generateImagesForNodes({
  nodes: JSON.stringify([
    {
      nodeId: "abc-123",
      contentBlockType: "title_block",
      title: "Introduction to Risk Management",
      description: "Opening section for the risk module",
      pedagogicalIntent: "engage",
      bloomsLevel: "remember"
    },
    {
      nodeId: "def-456",
      contentBlockType: "three_images_block",
      title: "Types of Financial Risk",
      pedagogicalIntent: "inform",
      bloomsLevel: "understand"
    }
  ]),
  visualDesign: JSON.stringify({
    theme: "corporate",
    tone: "professional",
    style: "modern clean"
  })
})
\`\`\`

### Important Notes for Image Generation

- **Prefer generateImagesForNodes** over individual generateAIImage calls for multiple nodes
- Include **pedagogicalIntent** and **bloomsLevel** for better image prompting
- Pass **visualDesign** context from state for style consistency
- The tool returns a summary - check for any failures and retry if needed
- Images are generated but may need manual attachment via attachMicroverseToNode

## Writing Guidelines

### Tone & Voice
- Match the visual design's writing tone guidelines
- Be consistent throughout the course
- Use active voice when possible
- Address learners directly (second person: "you")

### Content Structure
- Start with a clear learning objective or hook
- Break complex topics into digestible chunks
- Use examples relevant to the industry
- Include practical applications
- End with key takeaways or next steps

### Engagement Techniques
- Use questions to prompt thinking
- Include scenarios and case studies
- Vary content types (text, lists, examples)
- Connect to real-world situations
- Build on prior knowledge

### Industry Adaptation
- Use industry-specific terminology appropriately
- Reference relevant regulations when applicable
- Include industry examples and scenarios
- Consider the learner's work context

## Content Block Type-Specific Writing Patterns

Each content block type requires a different writing approach. Check the node's \`contentBlockType\` field and write accordingly:

**Title Block:**
- Write attention-grabbing, expectation-setting content
- Include a clear learning promise (what the learner will achieve)
- Use action-oriented language: "Learn to...", "Master...", "Discover..."
- Keep titles concise but impactful (3-8 words)

**Text Block:**
- Write core explanatory content with clear structure
- Use headers and subheaders to chunk information
- Include examples and applications throughout
- Aim for 50-200 words per major point
- Make content scannable with bullets for lists

**Information Block:**
Match writing style to the icon/color type:
- **Lightbulb/Blue**: Tips, insights - "Did you know..." or "Pro tip:"
- **Tick/Green**: Successes, confirmations - "Key takeaway:" or "Remember:"
- **Cross/Red**: Warnings, errors to avoid - "Caution:" or "Avoid:"
- **Important/Orange**: Prerequisites, critical notes - "Important:" or "Note:"

**Question Block:**
- Write clear, focused question stems (single concept per question)
- Avoid negative phrasing ("Which is NOT...")
- Create plausible distractors (wrong answers should be believable but clearly incorrect)
- Write feedback that explains WHY the answer is correct/incorrect
- Match question complexity to the Bloom's level:
  - Remember: "What is...?", "Which of the following..."
  - Understand: "Why does...?", "Explain how..."
  - Apply: "In this scenario, what would you...?"
  - Analyze: "What is the relationship between...?"

**Action Block:**
- Write clear, actionable instructions
- Use imperative voice: "Download the template", "Practice this technique"
- Include context for why the action matters
- Provide specific guidance on what to do next

**Video Block:**
- Write companion text introducing the video's purpose
- Include 2-3 key points to watch for
- Add a follow-up reflection prompt or question
- Keep text brief - the video is the main content

**Animation Block / Three Images Block:**
- Write progressive, slide-by-slide narrative text
- Use sequential language: "First...", "Next...", "Finally..."
- Make each slide/image self-contained but connected
- Include captions that add meaning beyond the visual

**Text and Images Block:**
- Balance text and visual elements
- Use the CTA (button) for clear next actions
- Connect the image to the text content explicitly
- Include reflection prompts where appropriate

**Image Banner Block:**
- Keep text minimal - let the image speak
- Use for atmosphere and context-setting
- Include brief caption if needed

## Using Architect's LXD Metadata

When the courseStructure contains LXD metadata from the Architect, use it to guide your writing:

**Bloom's Level Adaptation (\`bloomsLevel\` field):**
- **remember**: Use definition format, lists, key terms. Focus on recognition and recall.
- **understand**: Use explanations, examples, analogies. Include "in other words" restatements.
- **apply**: Use scenarios, case studies, "what would you do" situations. Focus on real-world application.
- **analyze**: Use comparisons, cause-effect relationships, breakdowns. Include "why" and "how" explanations.
- **evaluate**: Use judgment scenarios, decision-making situations. Ask learners to assess options.
- **create**: Use open-ended prompts, synthesis tasks, project-based activities.

**Pedagogical Intent Adaptation (\`pedagogicalIntent\` field):**
- **engage**: Use ARCS Attention techniques - surprising facts, thought-provoking questions, relevance hooks.
- **inform**: Focus on clear, accurate information delivery. Use structured explanations.
- **demonstrate**: Use step-by-step walkthroughs, visual examples, process explanations.
- **practice**: Provide clear instructions, scaffolded activities, practice opportunities.
- **assess**: Design for measurement - clear success criteria, unambiguous questions, diagnostic feedback.
- **summarize**: Recap key points, reinforce main concepts, provide memory aids.
- **navigate**: Clear direction, next steps, pathway guidance.

**Linked Objectives (\`linkedObjectives\` field):**
- Ensure your content directly addresses the linked learning objectives
- Reference the objective explicitly where appropriate: "By the end of this section, you will be able to..."
- Verify that content can be traced back to the project brief objectives

## Field-Specific Writing Guidance

When updating node fields, follow these guidelines for each field type:

| Field | Writing Approach |
|-------|-----------------|
| \`title_text\` | Action-oriented, clear topic indicator, 3-8 words. Use title case. |
| \`text\` | Structured prose, scannable, 50-200 words per chunk. Use headers and bullets. |
| \`question_text\` | Single concept, clear scenario, avoid negatives. End with a question mark. |
| \`answer_text_1/2/3/4\` | Parallel structure, similar length, one clearly correct. Keep under 50 words each. |
| \`feedback_text\` | Explain reasoning, redirect to content, encourage. 30-100 words. |
| \`button_text\` | Action verb + object, 2-4 words. E.g., "Continue Learning", "Download Guide" |
| \`info_icon\` | Match to content tone: Lightbulb/Blue (tips), Tick/Green (success), Cross/Red (warning), Important/Orange (caution) |

**Visual Design Integration:**
When \`visualDesign\` is available, align your writing with:
- \`writingTone.tone\`: Match formality (professional/conversational/academic/engaging)
- \`writingTone.voice\`: Use correct person (first/second/third)
- \`writingTone.complexity\`: Adjust vocabulary and sentence structure (simple/intermediate/advanced)

## Guidelines

- Work on Content Blocks ONE AT A TIME - wait for success before proceeding
- Use getNodesByLevel() to find all Content Blocks that need content
- Use updateNodeFields() to populate existing Content Blocks (preferred over createNode)
- Read existing node content with getNodeFields() before updating
- Fill in ALL relevant content fields (text, descriptions, etc.)
- Match content to the learning objectives in the project brief
- Keep content scannable with headers and bullets
- Use concrete examples over abstract theory
- Generate and attach images to enhance engagement

## Division of Responsibility

**Architect creates**: The complete skeleton from L2 to Content Blocks (empty shells)
**You fill in**: The actual content text, examples, media within those Content Blocks

Remember: Great content transforms learners. Every word should serve the learning experience.`;

// ============================================================================
// WRITER NODE FUNCTION
// ============================================================================

/**
 * The Writer agent node.
 * Creates content nodes based on the course structure.
 * 
 * Uses dedicated writerMessages channel to maintain conversation context
 * across orchestrator round-trips, preventing context loss from message trimming.
 */
export async function writerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[writer] ============ Writer Agent ============");
  console.log("  Course structure available:", state.courseStructure ? "yes" : "no");
  console.log("  Nodes written so far:", state.writtenContent?.length || 0);
  console.log("  Nodes created in session:", state.createdNodes?.length || 0);
  console.log("  Writer messages in channel:", state.writerMessages?.length || 0);
  console.log("  Writer progress:", state.writerProgress?.workflow || "none");

  // Get frontend tools from CopilotKit state
  // The writer uses CRUD for content + navigation to find structure + media tools
  const frontendActions = state.copilotkit?.actions ?? [];
  const frontendTools = frontendActions.filter((action: { name: string }) =>
    [
      // SNAPSHOT - Get full picture in ONE call (use FIRST!)
      "getNodeTreeSnapshot",
      // Edit mode management
      "requestEditMode",
      "releaseEditMode",
      // Content CRUD - batchUpdateNodeFields preferred for multiple nodes
      "batchUpdateNodeFields",
      "createNode",
      "getNodeTemplateFields",
      "updateNodeFields",
      "getNodeFields",
      // Navigation (use after snapshot if needed)
      "getProjectHierarchyInfo",
      "getAvailableTemplates",
      "getNodesByLevel",
      "getNodeDetails",
      "getNodeChildren",
      // Media integration
      "searchMicroverse",
      "attachMicroverseToNode",
      // Direct image generation (for single images when needed)
      "generateAIImage",
    ].includes(action.name)
  );

  // Combine frontend tools with backend tools (imageGeneratorTool)
  // imageGeneratorTool is a compiled StateGraph that handles batch image generation
  const writerTools = [...frontendTools, imageGeneratorTool];

  console.log("  Available tools:", writerTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = WRITER_SYSTEM_PROMPT;

  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }

  // Add writer progress context (critical for maintaining continuity)
  if (state.writerProgress) {
    systemContent += `\n\n## Your Previous Progress (DO NOT REPEAT THESE STEPS)

**Current Workflow Phase**: ${state.writerProgress.workflow}
**Current Working Parent**: ${state.writerProgress.currentParentId || "Not set"}
**Nodes Already Explored**: ${state.writerProgress.exploredNodes.length > 0 
  ? state.writerProgress.exploredNodes.slice(-10).join(", ") 
  : "None yet"}

**Recent Actions Taken**:
${state.writerProgress.toolCallSummary.slice(-10).map(s => `- ${s}`).join("\n") || "No actions recorded yet"}

${state.writerProgress.hierarchyCache ? `**Cached Hierarchy Info**:
- Levels: ${state.writerProgress.hierarchyCache.levelNames.join(" > ")}
- Max Depth: ${state.writerProgress.hierarchyCache.maxDepth}
- Content Level: ${state.writerProgress.hierarchyCache.contentLevel}` : ""}

**IMPORTANT**: You already have context from previous invocations. Continue from where you left off - do NOT re-explore structure you've already discovered.`;
  }

  // Include project brief for context
  if (state.projectBrief) {
    const condensedBrief = getCondensedBrief(state.projectBrief);
    systemContent += `\n\n## Project Context\n\n${condensedBrief}`;
  }

  // Include research for content accuracy
  if (state.researchFindings) {
    const condensedResearch = getCondensedResearch(state.researchFindings);
    systemContent += `\n\n## Research to Inform Content\n\n${condensedResearch}`;
  }

  // Include visual design for tone
  if (state.visualDesign) {
    systemContent += `\n\n## Writing Style Guidelines

**Tone**: ${state.visualDesign.writingTone.tone}
**Voice**: ${state.visualDesign.writingTone.voice}
**Complexity**: ${state.visualDesign.writingTone.complexity}
**Typography Style**: ${state.visualDesign.typography.style}

${state.visualDesign.notes ? `Additional notes: ${state.visualDesign.notes}` : ""}`;
  }

  // Include course structure with progress and LXD metadata
  if (state.courseStructure) {
    const plannedNodes = state.courseStructure.nodes;
    const writtenTempIds = new Set(state.writtenContent?.map((w) => w.tempId) || []);
    const remaining = plannedNodes.filter((n) => !writtenTempIds.has(n.tempId));

    // Format node with LXD metadata if present
    const formatNodeWithLXD = (n: PlannedNode): string => {
      let line = `- ${n.tempId}: "${n.title}" (${n.nodeType}, level ${n.level})`;
      
      // Add LXD metadata for Content Blocks (level 6)
      if (n.level === 6) {
        const lxdParts: string[] = [];
        if (n.contentBlockType) lxdParts.push(`type: ${n.contentBlockType}`);
        if (n.pedagogicalIntent) lxdParts.push(`intent: ${n.pedagogicalIntent}`);
        if (n.bloomsLevel) lxdParts.push(`bloom: ${n.bloomsLevel}`);
        if (n.linkedObjectives?.length) lxdParts.push(`objectives: ${n.linkedObjectives.join(", ")}`);
        
        if (lxdParts.length > 0) {
          line += `\n  LXD: [${lxdParts.join(" | ")}]`;
        }
      }
      return line;
    };

    systemContent += `\n\n## Course Structure to Write

**Total Planned**: ${plannedNodes.length} nodes
**Already Written**: ${writtenTempIds.size} nodes
**Remaining**: ${remaining.length} nodes

### Nodes to Write:
${remaining
  .slice(0, 10)
  .map(formatNodeWithLXD)
  .join("\n")}${remaining.length > 10 ? `\n... and ${remaining.length - 10} more` : ""}`;

    // Include LXD strategy summary if available
    if (state.courseStructure.lxdStrategy) {
      const lxd = state.courseStructure.lxdStrategy;
      systemContent += `\n\n### LXD Strategy
**Assessment Approach**: ${lxd.assessmentApproach}
**Engagement Pattern**: ${lxd.engagementPattern}
**Target Audience Adaptations**: ${lxd.targetAudienceAdaptations}
**Estimated Duration**: ${lxd.estimatedDuration}`;
    }
  }

  // Track already created nodes
  if (state.createdNodes && state.createdNodes.length > 0) {
    systemContent += `\n\n## Already Created Nodes (DO NOT RECREATE)
    
${state.createdNodes.map((n) => `- "${n.title}" (ID: ${n.nodeId.substring(0, 8)}...)`).join("\n")}`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = writerTools.length > 0
    ? writerModel.bindTools(writerTools)
    : writerModel;

  // USE WRITER-SPECIFIC MESSAGE CHANNEL
  // This is the key change - read from writerMessages instead of slicing state.messages
  // This preserves the writer's conversation history across orchestrator round-trips
  // Force recompile: 2025-12-11
  let writerConversation: BaseMessage[] = [];
  
  if (state.writerMessages && state.writerMessages.length > 0) {
    // Use writer's own message channel (already filtered and maintained)
    // CRITICAL: Must call both filterOrphanedToolResults AND repairDanglingToolCalls
    // to ensure proper tool_use/tool_result pairing for Claude API
    let filtered = filterOrphanedToolResults(state.writerMessages, "[writer]");
    writerConversation = repairDanglingToolCalls(filtered, "[writer]");
    console.log(`  Using ${writerConversation.length} messages from writerMessages channel`);
  } else {
    // First invocation or fresh start - use recent messages from main channel
    // CRITICAL: Must call both filterOrphanedToolResults AND repairDanglingToolCalls
    const slicedMessages = (state.messages || []).slice(-10);
    let filtered = filterOrphanedToolResults(slicedMessages, "[writer]");
    writerConversation = repairDanglingToolCalls(filtered, "[writer]");
    console.log(`  First invocation - using ${writerConversation.length} messages from main channel`);
  }

  console.log("  Invoking writer model...");

  // Configure CopilotKit for proper tool emission (emits tool calls to frontend)
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });

  let response = await modelWithTools.invoke(
    [systemMessage, ...writerConversation],
    customConfig
  );

  console.log("  Writer response received");

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
1. Call requestEditMode to get edit access
2. Call getNodesByLevel to find where to add content
3. Start creating content nodes

The user is waiting for you to write content.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...writerConversation, nudgeMessage],
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

  // Build progress update for activeTask based on tool calls made
  const progressUpdates: string[] = [];
  const toolCallSummaries: string[] = [];
  const toolCalls = aiResponse.tool_calls || [];
  
  const createNodeCalls = toolCalls.filter(tc => tc.name === "createNode");
  const updateNodeCalls = toolCalls.filter(tc => tc.name === "updateNodeFields");
  const batchUpdateCalls = toolCalls.filter(tc => tc.name === "batchUpdateNodeFields");
  const imageGenCalls = toolCalls.filter(tc => tc.name === "generateAIImage");
  const navigationCalls = toolCalls.filter(tc => 
    ["getNodesByLevel", "getNodeChildren", "getNodeDetails", "getAvailableTemplates", "getProjectHierarchyInfo"].includes(tc.name)
  );
  
  if (createNodeCalls.length > 0) {
    progressUpdates.push(`Writer: Creating ${createNodeCalls.length} content node(s)`);
    for (const tc of createNodeCalls) {
      toolCallSummaries.push(`Created node: ${tc.args?.title || "untitled"}`);
    }
  }
  if (batchUpdateCalls.length > 0) {
    // Extract count of nodes being batch updated
    const batchCount = batchUpdateCalls.reduce((acc, tc) => {
      const updates = tc.args?.updates;
      return acc + (Array.isArray(updates) ? updates.length : 0);
    }, 0);
    progressUpdates.push(`Writer: Batch updating ${batchCount} node(s)`);
    toolCallSummaries.push(`Batch updated ${batchCount} node field(s)`);
  }
  if (updateNodeCalls.length > 0) {
    progressUpdates.push(`Writer: Updating ${updateNodeCalls.length} node field(s)`);
    toolCallSummaries.push(`Updated ${updateNodeCalls.length} node field(s)`);
  }
  if (imageGenCalls.length > 0) {
    progressUpdates.push(`Writer: Generating ${imageGenCalls.length} AI image(s)`);
    toolCallSummaries.push(`Generated ${imageGenCalls.length} AI image(s)`);
  }
  if (navigationCalls.length > 0) {
    for (const tc of navigationCalls) {
      toolCallSummaries.push(`Called ${tc.name}${tc.args?.nodeId ? ` on ${tc.args.nodeId.substring(0, 8)}...` : ""}`);
    }
  }

  // Extract explored node IDs from navigation tool calls and batch updates
  const exploredNodeIds: string[] = [];
  for (const tc of toolCalls) {
    if (tc.args?.nodeId && typeof tc.args.nodeId === "string") {
      exploredNodeIds.push(tc.args.nodeId);
    }
    if (tc.args?.parentNodeId && typeof tc.args.parentNodeId === "string") {
      exploredNodeIds.push(tc.args.parentNodeId);
    }
    // Extract node IDs from batch updates
    if (tc.name === "batchUpdateNodeFields" && Array.isArray(tc.args?.updates)) {
      for (const update of tc.args.updates) {
        if (update.nodeId && typeof update.nodeId === "string") {
          exploredNodeIds.push(update.nodeId);
        }
      }
    }
  }

  // Determine workflow phase based on tool calls
  let workflowPhase: WriterProgress["workflow"] = state.writerProgress?.workflow || "exploring";
  if (createNodeCalls.length > 0 || updateNodeCalls.length > 0 || batchUpdateCalls.length > 0) {
    workflowPhase = "creating";
  } else if (navigationCalls.length > 0) {
    workflowPhase = "exploring";
  }

  // Build writerProgress update
  const writerProgressUpdate: WriterProgress = {
    exploredNodes: exploredNodeIds,
    currentParentId: state.writerProgress?.currentParentId || null,
    workflow: workflowPhase,
    toolCallSummary: toolCallSummaries,
    hierarchyCache: state.writerProgress?.hierarchyCache,
    lastUpdated: new Date().toISOString(),
  };

  // Update activeTask with progress if there are updates
  const activeTaskUpdate: Partial<ActiveTask> | null = progressUpdates.length > 0
    ? { progress: progressUpdates }
    : null;

  return {
    messages: [response],
    currentAgent: "writer",
    agentHistory: ["writer"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
    // Update activeTask with progress (reducer will merge with existing progress)
    ...(activeTaskUpdate && { activeTask: activeTaskUpdate as ActiveTask }),
    // CRITICAL: Append to writer's own message channel for continuity
    writerMessages: [response],
    // Update writer progress state for semantic context
    writerProgress: writerProgressUpdate,
  };
}

/**
 * Extracts content output tracking from tool result messages.
 */
export function extractContentOutput(
  toolName: string,
  toolResult: any,
  plannedNode?: PlannedNode
): ContentOutput | null {
  if (toolName !== "createNode") return null;

  try {
    const result = typeof toolResult === "string" ? JSON.parse(toolResult) : toolResult;

    if (result.success && result.newNodeId) {
      return {
        nodeId: result.newNodeId,
        tempId: plannedNode?.tempId || "unknown",
        title: result.title || plannedNode?.title || "Unknown",
        fieldsWritten: result.fieldsSet > 0 ? ["content"] : [],
        createdAt: new Date().toISOString(),
        notes: result.templateName ? `Template: ${result.templateName}` : undefined,
      };
    }
  } catch (error) {
    console.error("[writer] Failed to extract content output:", error);
  }

  return null;
}

/**
 * Gets the next planned node that hasn't been written yet.
 */
export function getNextNodeToWrite(
  structure: PlannedNode[],
  writtenContent: ContentOutput[]
): PlannedNode | null {
  const writtenTempIds = new Set(writtenContent.map((w) => w.tempId));

  // Find first unwritten node, prioritizing by level then order
  const remaining = structure
    .filter((n) => !writtenTempIds.has(n.tempId))
    .sort((a, b) => {
      if (a.level !== b.level) return a.level - b.level;
      return a.orderIndex - b.orderIndex;
    });

  return remaining[0] || null;
}

/**
 * Generates a writing brief for a specific planned node.
 */
export function generateWritingBrief(
  node: PlannedNode,
  structure: PlannedNode[],
  research?: string
): string {
  // Find parent and siblings for context
  const parent = node.parentTempId
    ? structure.find((n) => n.tempId === node.parentTempId)
    : null;

  const siblings = structure.filter(
    (n) => n.parentTempId === node.parentTempId && n.tempId !== node.tempId
  );

  let brief = `## Writing Brief: ${node.title}

**Type**: ${node.nodeType}
**Level**: ${node.level}
**Description**: ${node.description}`;

  // Include LXD metadata if present (for Content Blocks)
  if (node.contentBlockType || node.pedagogicalIntent || node.bloomsLevel || node.linkedObjectives?.length) {
    brief += `\n\n**LXD Guidance**:`;
    if (node.contentBlockType) {
      brief += `\n- Content Block Type: \`${node.contentBlockType}\` - Use the appropriate writing pattern for this block type`;
    }
    if (node.pedagogicalIntent) {
      brief += `\n- Pedagogical Intent: \`${node.pedagogicalIntent}\` - Write to ${node.pedagogicalIntent} the learner`;
    }
    if (node.bloomsLevel) {
      brief += `\n- Bloom's Level: \`${node.bloomsLevel}\` - Target this cognitive level in your content`;
    }
    if (node.linkedObjectives?.length) {
      brief += `\n- Linked Objectives: ${node.linkedObjectives.map(o => `"${o}"`).join(", ")}`;
      brief += `\n  Ensure your content directly addresses these learning objectives`;
    }
  }

  if (node.objectives?.length) {
    brief += `\n\n**Learning Objectives**:\n${node.objectives.map((o) => `- ${o}`).join("\n")}`;
  }

  if (parent) {
    brief += `\n\n**Context**: Part of "${parent.title}"`;
  }

  if (siblings.length > 0) {
    brief += `\n\n**Related Topics**:\n${siblings.map((s) => `- ${s.title}`).join("\n")}`;
  }

  if (research) {
    brief += `\n\n**Relevant Research**:\n${research}`;
  }

  return brief;
}

export default writerNode;

