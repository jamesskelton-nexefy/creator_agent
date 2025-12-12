/**
 * Media Agent
 *
 * Specialized agent for media library (Microverse) operations.
 * Handles searching, viewing, generating AI images, and attaching media assets to content nodes.
 *
 * Tools (Frontend):
 * - searchMicroverse - Search media assets
 * - getMicroverseDetails - Get detailed asset info
 * - getMicroverseUsage - Check where an asset is used
 * - generateMicroverseAssets - Generate new assets from images
 * - generateAIImage - Generate AI images using nano-banana-pro
 * - attachMicroverseToNode - Attach media to a node field
 * - detachMicroverseFromNode - Remove media from a node
 * - getNodeMicroverseFields - List media fields on a node
 *
 * Output: Responds directly to user with media information
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
import type { OrchestratorState } from "../state/agent-state";
import { generateTaskContext } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  stripThinkingBlocks,
  hasUsableResponse,
} from "../utils";

// Import tool categories for reference
import { TOOL_CATEGORIES } from "../orchestrator-agent-deep";

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const mediaAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 4000,
  temperature: 0.4,
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const MEDIA_AGENT_SYSTEM_PROMPT = `You are The Media Agent - a specialized agent for the Microverse media library.

## Your Role

You help users find, manage, and attach media assets (images, videos, documents, 3D models) to content nodes. The Microverse is the central media library.

## Your Tools

### Media Discovery
- **searchMicroverse(searchTerm?, fileType?, category?, limit?)** - Search media assets
  - fileType: "image", "video", "document", "3d"
  - Returns thumbnails, metadata, and file info
  - Empty search returns all assets

- **getMicroverseDetails(fileId)** - Get detailed info about an asset
  - Shows versions, metadata, storage path
  - Returns signed URLs for viewing

- **getMicroverseUsage(fileId)** - Check where an asset is used
  - Shows all nodes that reference this asset
  - Helps avoid orphaned or duplicate attachments

### Media Management
- **generateMicroverseAssets(filename, title?, description?, tags?)** - Create new asset
  - Uploads from /images/ folder to the library
  - Registers metadata in the database

### AI Image Generation
- **generateAIImage(prompt, preset?, aspectRatio?, title?, description?, tags?)** - Generate AI images
  - Uses Google's nano-banana-pro model via Replicate
  - **Presets for eLearning:**
    - "banner" (21:9) - Course/module banners and headers
    - "hero" (16:9) - Hero images and presentation slides
    - "content" (16:9) - General content illustrations
    - "thumbnail" (3:2) - Card thumbnails and previews
    - "square" (1:1) - Icons, avatars, social media
    - "portrait" (3:4) - Character portraits, person photos
    - "custom" - Use custom aspectRatio parameter
  - Automatically stores generated images in Microverse
  - Deduplicates identical images by hash

### Attaching Media to Nodes
- **getNodeMicroverseFields(nodeId?)** - List media fields on a node
  - Shows which fields can accept media
  - Shows current attachments
  - Returns field assignment IDs needed for attach/detach

- **attachMicroverseToNode(nodeId, fieldKey, fileId)** - Attach media to node
  - Requires edit mode to be active
  - Links asset to a specific field on the node

- **detachMicroverseFromNode(nodeId, fieldKey)** - Remove media from node
  - Requires edit mode to be active
  - Clears the media attachment from the field

## Workflow Examples

### Finding Media
User: "Find images of trucks"
→ Call searchMicroverse({ searchTerm: "truck", fileType: "image" })

### Checking Asset Usage
User: "Is this image used anywhere?"
→ Call getMicroverseUsage({ fileId: "..." })

### Attaching Media to Node
1. Call getNodeMicroverseFields(nodeId) to see available fields
2. Ensure edit mode is active (handled by node agent)
3. Call attachMicroverseToNode(nodeId, fieldKey, fileId)

### Viewing All Media
User: "Show me all videos in the library"
→ Call searchMicroverse({ fileType: "video" })

### Generating AI Images
User: "Create a banner image of a modern office with people collaborating"
→ Call generateAIImage({ 
    prompt: "Modern open office space with diverse professionals collaborating around a whiteboard, natural lighting, warm colors, professional photography style",
    preset: "banner",
    title: "Collaboration Banner",
    description: "AI generated office collaboration scene"
  })

User: "Generate a thumbnail showing workplace safety"
→ Call generateAIImage({
    prompt: "Workplace safety equipment including hard hat, safety vest, and protective glasses on a clean background, professional product photography",
    preset: "thumbnail",
    title: "Safety Equipment Thumbnail"
  })

## Communication Style

- Show thumbnails/previews when available
- Summarize search results (e.g., "Found 12 images matching 'safety'")
- Note which assets are already in use
- Warn before detaching widely-used assets

## When to Hand Back

Include \`[DONE]\` in your response when:
- You've shown the requested media assets
- You've completed the attachment/detachment
- The user wants to do something outside media management

Example: "I've attached the safety diagram to the lesson node. [DONE]"`;

// ============================================================================
// MEDIA AGENT NODE FUNCTION
// ============================================================================

/**
 * The Media Agent node.
 * Handles Microverse media library operations.
 */
export async function mediaAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[media-agent] ============ Media Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Media tools
  const mediaToolNames = TOOL_CATEGORIES.media;
  
  const mediaAgentTools = frontendActions.filter((action: { name: string }) =>
    mediaToolNames.includes(action.name)
  );

  console.log("  Available tools:", mediaAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", mediaAgentTools.length);

  // Build system prompt with task context
  let systemContent = MEDIA_AGENT_SYSTEM_PROMPT;
  
  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = mediaAgentTools.length > 0
    ? mediaAgentModel.bindTools(mediaAgentTools)
    : mediaAgentModel;

  // Filter messages for this agent's context
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-10);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[media-agent]");

  console.log("  Invoking media agent model...");

  // Configure CopilotKit for proper tool emission (emits tool calls to frontend)
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    customConfig
  );

  console.log("  Media agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. If searching for media, call searchMicroverse
2. If checking media details, call getMicroverseDetails
3. Provide helpful results showing what you found

The user is waiting for media library information.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...recentMessages, nudgeMessage],
      customConfig
    );
    
    aiResponse = response as AIMessage;
    
    if (hasUsableResponse(aiResponse)) {
      console.log("  [RETRY] Success - got usable response on retry");
    }
  }

  return {
    messages: [response],
    currentAgent: "media_agent",
    agentHistory: ["media_agent"],
    routingDecision: null,
  };
}

export default mediaAgentNode;






