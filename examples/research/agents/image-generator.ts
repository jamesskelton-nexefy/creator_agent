/**
 * Image Generator Subagent
 *
 * A compiled StateGraph wrapped as a LangChain tool for generating and attaching
 * images to content blocks. This pattern ensures CopilotKit compatibility by
 * encapsulating all image generation internals.
 *
 * Architecture:
 * - Compiled StateGraph handles the image generation workflow
 * - Tool wrapper invokes the graph and returns a summary
 * - Internal tool calls (generateImage, attachMicroverse) are hidden from CopilotKit
 *
 * Tools used internally:
 * - generateImage - AI image generation via Replicate
 * - Node field updates for attaching images
 *
 * Input: JSON arrays of nodes needing images + visual design context
 * Output: Summary of generated/attached images for Writer
 */

import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { SystemMessage, AIMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { generateImageTool } from "../lib/imageTools";
import type { ImageNodeInput, ImageGenerationResult, ImageGeneratorState, BloomsLevel, PedagogicalIntent } from "../state/agent-state";

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const IMAGE_GENERATOR_SYSTEM_PROMPT = `You are an Image Generator specialized in creating visuals for eLearning content. Your task is to generate and attach appropriate images to content blocks.

## Your Tools

You have access to these tools:
- **generateImage** - Generate an AI image with a prompt and preset. Returns image data including a base64 buffer and URL.
- **recordImageResult** - Record the result of image generation for a node (call after generateImage succeeds or fails).

## Content Block Image Requirements

| Block Type | Required | Preset | Style Guidance |
|------------|----------|--------|----------------|
| title_block | Yes | hero or banner | Engaging, sets tone for section |
| three_images_block | Yes (x3) | content | Related trio, consistent style |
| image_banner_block | Yes | banner (21:9) | Atmospheric, wide format |
| text_and_images_block | Yes | content | Contextual, supports text |
| question_block | Optional | content | Scenario-based, situational |
| video_block | Optional | thumbnail | Preview/poster frame |

## Image Presets Reference

- **banner** (21:9) - Wide course banners and headers
- **hero** (16:9) - Hero images and slides
- **content** (16:9) - General content illustrations
- **thumbnail** (3:2) - Card thumbnails
- **square** (1:1) - Icons and avatars
- **portrait** (3:4) - Portrait images of people

## LXD-Informed Image Prompting

Adapt image style based on pedagogical metadata:

**By pedagogicalIntent:**
- "engage" -> Dynamic, attention-grabbing, surprising elements, vibrant colors
- "inform" -> Clear, explanatory, diagram-like, clean compositions
- "demonstrate" -> Step-by-step, process-focused, instructional
- "practice" -> Interactive feel, hands-on scenarios, workplace settings
- "assess" -> Neutral, unbiased, scenario-setting, professional

**By bloomsLevel:**
- "remember" -> Simple, iconic, memorable visuals with clear subjects
- "understand" -> Explanatory diagrams, concept illustrations
- "apply" -> Real-world workplace scenarios, practical situations
- "analyze" -> Comparison visuals, cause-effect imagery, side-by-side
- "evaluate" -> Decision-point scenarios, multiple perspectives
- "create" -> Open-ended, inspirational, creative prompts

## Visual Design Consistency

Reference the provided visual design context:
- Match the theme (e.g., "corporate", "playful", "technical")
- Align with tone (e.g., "professional", "friendly", "serious")
- Maintain style consistency across all generated images
- Use consistent color palettes and visual language

## Workflow

For each node in the task:
1. Analyze the contentBlockType to determine required images and preset
2. Read the title and description for content context
3. Consider pedagogicalIntent and bloomsLevel for style adaptation
4. Craft an appropriate, detailed image prompt
5. Call generateImage with the correct preset
6. Call recordImageResult to log the outcome
7. Continue to the next node

## Image Prompt Best Practices

- Be specific about visual style: "professional photography", "flat illustration", "modern 3D render"
- Include context details: setting, lighting, mood, color palette
- Describe composition: "centered subject", "wide angle view", "close-up detail"
- Specify what NOT to include: "no text", "no people", "simple background"
- Match the eLearning context: "corporate training environment", "professional workplace"

## Example Prompts by Block Type

**title_block (hero)**:
"Professional corporate photography, modern open-plan office with natural lighting, team collaboration scene, warm professional atmosphere, no text overlay, shallow depth of field, 16:9 aspect ratio"

**three_images_block (content x3)**:
1. "Clean flat illustration of risk assessment checklist, blue corporate colors, minimal design, white background"
2. "Flat illustration of data analysis dashboard, matching blue palette, professional icons"
3. "Flat illustration of team meeting discussing results, consistent style with previous images"

**question_block (content)**:
"Professional workplace scenario, employee at computer reviewing documents, neutral expression, modern office setting, realistic photography style"

Process all nodes efficiently. Generate high-quality, contextually appropriate images that enhance the learning experience.`;

// ============================================================================
// STATE ANNOTATION
// ============================================================================

/**
 * Messages reducer that accumulates messages.
 */
function messagesReducer(existing: BaseMessage[], update: BaseMessage[]): BaseMessage[] {
  return [...(existing || []), ...(update || [])];
}

/**
 * Results reducer that accumulates image generation results.
 */
function resultsReducer(existing: ImageGenerationResult[], update: ImageGenerationResult[]): ImageGenerationResult[] {
  return [...(existing || []), ...(update || [])];
}

/**
 * State annotation for the Image Generator graph.
 */
const ImageGeneratorStateAnnotation = Annotation.Root({
  /** Input: nodes to process */
  nodesToProcess: Annotation<ImageNodeInput[]>({
    reducer: (_, update) => update,
    default: () => [],
  }),
  /** Input: visual design context */
  visualDesign: Annotation<{ theme?: string; tone?: string; style?: string }>({
    reducer: (_, update) => update,
    default: () => ({}),
  }),
  /** Messages for the LLM agent loop */
  messages: Annotation<BaseMessage[]>({
    reducer: messagesReducer,
    default: () => [],
  }),
  /** Output: results of image generation */
  results: Annotation<ImageGenerationResult[]>({
    reducer: resultsReducer,
    default: () => [],
  }),
  /** Output: summary for Writer */
  summary: Annotation<string>({
    reducer: (_, update) => update,
    default: () => "",
  }),
});

type ImageGeneratorStateType = typeof ImageGeneratorStateAnnotation.State;

// ============================================================================
// INTERNAL TOOLS
// ============================================================================

/**
 * Tool for recording image generation results.
 * This allows the agent to track what was generated for each node.
 */
const recordImageResultTool = tool(
  async ({ nodeId, contentBlockType, success, fileId, prompt, preset, error }): Promise<string> => {
    console.log(`[recordImageResult] Node ${nodeId}: ${success ? "SUCCESS" : "FAILED"}`);
    return JSON.stringify({
      recorded: true,
      nodeId,
      contentBlockType,
      success,
      fileId: fileId || null,
      prompt,
      preset,
      error: error || null,
    });
  },
  {
    name: "recordImageResult",
    description: "Record the result of image generation for a node. Call this after generateImage succeeds or fails.",
    schema: z.object({
      nodeId: z.string().describe("The node ID this image was generated for"),
      contentBlockType: z.string().describe("The content block type"),
      success: z.boolean().describe("Whether image generation succeeded"),
      fileId: z.string().optional().describe("The file ID if generation succeeded"),
      prompt: z.string().describe("The prompt used for generation"),
      preset: z.string().describe("The preset used (banner, hero, content, etc.)"),
      error: z.string().optional().describe("Error message if generation failed"),
    }),
  }
);

// Tools available to the image generator
const imageGeneratorTools = [generateImageTool, recordImageResultTool];

// ============================================================================
// GRAPH NODES
// ============================================================================

/**
 * Image generator node - calls the LLM with tools.
 */
async function imageGeneratorNode(state: ImageGeneratorStateType): Promise<Partial<ImageGeneratorStateType>> {
  console.log(`[imageGeneratorNode] Processing ${state.nodesToProcess.length} nodes...`);
  
  const model = new ChatAnthropic({
    model: "claude-sonnet-4-20250514",
    temperature: 0.7,
    maxTokens: 8000,
  }).bindTools(imageGeneratorTools);

  // Build system message with context
  const nodesContext = state.nodesToProcess.map((n, i) => 
    `${i + 1}. nodeId: "${n.nodeId}", type: ${n.contentBlockType}, title: "${n.title}"` +
    (n.description ? `, description: "${n.description}"` : "") +
    (n.pedagogicalIntent ? `, intent: ${n.pedagogicalIntent}` : "") +
    (n.bloomsLevel ? `, bloom: ${n.bloomsLevel}` : "")
  ).join("\n");

  const designContext = state.visualDesign.theme || state.visualDesign.tone || state.visualDesign.style
    ? `Theme: ${state.visualDesign.theme || "not specified"}, Tone: ${state.visualDesign.tone || "not specified"}, Style: ${state.visualDesign.style || "not specified"}`
    : "No specific visual design context provided - use professional corporate style.";

  const taskMessage = `## Current Task

Generate images for these ${state.nodesToProcess.length} content blocks:

${nodesContext}

## Visual Design Context
${designContext}

Process each node in order. For each:
1. Determine the appropriate preset based on contentBlockType
2. Craft a detailed image prompt considering the title, description, and LXD metadata
3. Call generateImage with your prompt and preset
4. Call recordImageResult to log the outcome

Start with the first node now.`;

  const systemMessage = new SystemMessage({
    content: IMAGE_GENERATOR_SYSTEM_PROMPT,
  });

  // If this is the first invocation, add the task as a human message
  let messagesToSend: BaseMessage[] = [systemMessage];
  if (state.messages.length === 0) {
    messagesToSend.push(new AIMessage({ content: "I'll generate images for the content blocks. Let me process each one." }));
    messagesToSend.push(new SystemMessage({ content: taskMessage }));
  } else {
    messagesToSend = [systemMessage, ...state.messages];
  }

  const response = await model.invoke(messagesToSend);
  
  console.log(`[imageGeneratorNode] Response received, tool_calls: ${(response as AIMessage).tool_calls?.length || 0}`);
  
  return { messages: [response] };
}

/**
 * Tool execution node.
 */
const toolNode = new ToolNode(imageGeneratorTools);

/**
 * Determines whether to continue to tools or end.
 */
function shouldContinue(state: ImageGeneratorStateType): "tools" | "summarize" {
  const lastMessage = state.messages[state.messages.length - 1];
  
  if (lastMessage && "tool_calls" in lastMessage) {
    const aiMessage = lastMessage as AIMessage;
    if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
      return "tools";
    }
  }
  
  return "summarize";
}

/**
 * Summarize node - extracts results and builds summary.
 */
function summarizeNode(state: ImageGeneratorStateType): Partial<ImageGeneratorStateType> {
  console.log(`[summarizeNode] Building summary from ${state.messages.length} messages...`);
  
  // Extract results from recordImageResult tool calls
  const results: ImageGenerationResult[] = [];
  
  for (const message of state.messages) {
    if (message instanceof ToolMessage && message.name === "recordImageResult") {
      try {
        const content = typeof message.content === "string" 
          ? JSON.parse(message.content)
          : message.content;
        
        if (content.recorded) {
          results.push({
            nodeId: content.nodeId,
            contentBlockType: content.contentBlockType,
            fileId: content.fileId || "",
            prompt: content.prompt,
            preset: content.preset,
            success: content.success,
            error: content.error,
          });
        }
      } catch (e) {
        console.log(`[summarizeNode] Failed to parse tool result: ${e}`);
      }
    }
  }
  
  const successCount = results.filter(r => r.success).length;
  const totalCount = state.nodesToProcess.length;
  
  const summary = `Image generation complete: ${successCount}/${totalCount} images generated.\n` +
    results.map(r => 
      `- ${r.nodeId} (${r.contentBlockType}): ${r.success ? "OK" : "FAILED - " + (r.error || "unknown error")}`
    ).join("\n");
  
  console.log(`[summarizeNode] Summary: ${successCount}/${totalCount} successful`);
  
  return { results, summary };
}

// ============================================================================
// GRAPH DEFINITION
// ============================================================================

/**
 * Build and compile the Image Generator graph.
 */
const imageGeneratorBuilder = new StateGraph(ImageGeneratorStateAnnotation)
  .addNode("generate", imageGeneratorNode)
  .addNode("tools", toolNode)
  .addNode("summarize", summarizeNode)
  .addEdge(START, "generate")
  .addConditionalEdges("generate", shouldContinue, {
    tools: "tools",
    summarize: "summarize",
  })
  .addEdge("tools", "generate")
  .addEdge("summarize", END);

export const imageGeneratorGraph = imageGeneratorBuilder.compile();

// ============================================================================
// TOOL WRAPPER
// ============================================================================

/**
 * Tool wrapper for the Image Generator graph.
 * This encapsulates the entire image generation workflow as a single tool call.
 * 
 * CopilotKit Compatibility:
 * - Only this wrapper tool is visible to CopilotKit
 * - Internal generateImage and recordImageResult calls are hidden
 * - Returns a clean text summary for the Writer agent
 */
export const imageGeneratorTool = tool(
  async ({ nodes, visualDesign }): Promise<string> => {
    console.log(`[imageGeneratorTool] Starting image generation...`);
    
    // Parse inputs
    let nodesToProcess: ImageNodeInput[];
    let designContext: { theme?: string; tone?: string; style?: string };
    
    try {
      nodesToProcess = typeof nodes === "string" ? JSON.parse(nodes) : nodes;
      designContext = typeof visualDesign === "string" ? JSON.parse(visualDesign) : visualDesign;
    } catch (e) {
      return `Error parsing inputs: ${e instanceof Error ? e.message : "Invalid JSON"}`;
    }
    
    if (!nodesToProcess || nodesToProcess.length === 0) {
      return "No nodes provided for image generation.";
    }
    
    console.log(`[imageGeneratorTool] Processing ${nodesToProcess.length} nodes...`);
    console.log(`[imageGeneratorTool] Visual design: ${JSON.stringify(designContext)}`);
    
    try {
      // Invoke the compiled graph
      const result = await imageGeneratorGraph.invoke({
        nodesToProcess,
        visualDesign: designContext,
        messages: [],
        results: [],
        summary: "",
      });
      
      console.log(`[imageGeneratorTool] Graph complete. Results: ${result.results?.length || 0}`);
      
      // Return the summary
      return result.summary || "Image generation completed but no summary was generated.";
    } catch (error) {
      console.error(`[imageGeneratorTool] Graph execution failed:`, error);
      return `Image generation failed: ${error instanceof Error ? error.message : "Unknown error"}`;
    }
  },
  {
    name: "generateImagesForNodes",
    description: `Generate and attach images to content blocks. Call this after writing content to nodes that require images.

Content blocks requiring images:
- title_block: Hero/banner image (required)
- three_images_block: Three related images (required)
- image_banner_block: Wide banner image (required)
- text_and_images_block: Contextual image (required)
- question_block: Scenario image (optional, if scenario-based)
- video_block: Thumbnail/poster (optional)

Pass the node details and visual design context. The tool will generate appropriate images.

Note: Images are generated but NOT automatically attached to nodes. The returned summary includes file IDs that can be used with attachMicroverseToNode if needed.`,
    schema: z.object({
      nodes: z.string().describe(
        "JSON array of nodes needing images. Each node: {nodeId, contentBlockType, title, description?, pedagogicalIntent?, bloomsLevel?}"
      ),
      visualDesign: z.string().describe(
        "JSON object with visual design context: {theme?, tone?, style?}"
      ),
    }),
  }
);

// ============================================================================
// EXPORTS
// ============================================================================

export default imageGeneratorTool;


