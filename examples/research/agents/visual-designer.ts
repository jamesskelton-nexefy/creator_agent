/**
 * Visual Designer Agent
 *
 * Defines course aesthetics - fonts, colors, branding, and writing tone.
 * Presents design options for user selection.
 * Can research design trends and find existing brand assets.
 *
 * Tools:
 * - offerOptions - Present design choices to user
 * - web_search - Research design trends, industry standards, color psychology
 * - searchMicroverse - Find existing brand assets, logos, images
 * - getMicroverseDetails - Get asset information
 *
 * Input: Reads projectBrief from state
 * Output: visualDesign spec in shared state
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState, VisualDesign, AgentWorkState, VisualDesignerPhase, ActiveTask } from "../state/agent-state";
import { getCondensedBrief, VISUAL_DESIGNER_PHASES, generateTaskContext } from "../state/agent-state";
import { researcherTools } from "./researcher";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  hasUsableResponse,
} from "../utils";

// Extract just the web_search tool for the designer
const webSearchTool = researcherTools.find((t) => t.name === "web_search");

// Message filtering now handled by centralized utils/context-management.ts

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const designerModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// DESIGN PRESETS
// ============================================================================

/**
 * Pre-defined design themes for different contexts.
 */
export const DESIGN_THEMES = {
  corporate: {
    name: "Corporate Professional",
    description: "Clean, professional design suitable for business training",
    colors: {
      primary: "#1e40af", // Deep blue
      secondary: "#64748b", // Slate
      accent: "#0ea5e9", // Sky blue
      background: "#f8fafc",
      text: "#1e293b",
    },
    typography: {
      headingFont: "Inter",
      bodyFont: "Open Sans",
      style: "formal" as const,
    },
    writingTone: {
      tone: "professional" as const,
      voice: "second-person" as const,
      complexity: "intermediate" as const,
    },
  },
  healthcare: {
    name: "Healthcare & Safety",
    description: "Trust-inspiring design for medical and safety training",
    colors: {
      primary: "#0d9488", // Teal
      secondary: "#6b7280", // Gray
      accent: "#10b981", // Emerald
      background: "#f0fdfa",
      text: "#134e4a",
    },
    typography: {
      headingFont: "Lato",
      bodyFont: "Source Sans Pro",
      style: "formal" as const,
    },
    writingTone: {
      tone: "professional" as const,
      voice: "second-person" as const,
      complexity: "intermediate" as const,
    },
  },
  technical: {
    name: "Technical & Engineering",
    description: "Modern tech-focused design for technical training",
    colors: {
      primary: "#7c3aed", // Violet
      secondary: "#4b5563", // Gray
      accent: "#8b5cf6", // Purple
      background: "#faf5ff",
      text: "#1f2937",
    },
    typography: {
      headingFont: "JetBrains Mono",
      bodyFont: "Inter",
      style: "technical" as const,
    },
    writingTone: {
      tone: "professional" as const,
      voice: "second-person" as const,
      complexity: "advanced" as const,
    },
  },
  hospitality: {
    name: "Hospitality & Service",
    description: "Warm, welcoming design for customer service training",
    colors: {
      primary: "#ea580c", // Orange
      secondary: "#78716c", // Stone
      accent: "#f59e0b", // Amber
      background: "#fffbeb",
      text: "#292524",
    },
    typography: {
      headingFont: "Playfair Display",
      bodyFont: "Lora",
      style: "friendly" as const,
    },
    writingTone: {
      tone: "conversational" as const,
      voice: "second-person" as const,
      complexity: "simple" as const,
    },
  },
  education: {
    name: "Education & Academic",
    description: "Classic academic style for educational institutions",
    colors: {
      primary: "#dc2626", // Red
      secondary: "#6b7280", // Gray
      accent: "#2563eb", // Blue
      background: "#fef2f2",
      text: "#1f2937",
    },
    typography: {
      headingFont: "Merriweather",
      bodyFont: "Georgia",
      style: "formal" as const,
    },
    writingTone: {
      tone: "academic" as const,
      voice: "third-person" as const,
      complexity: "advanced" as const,
    },
  },
  modern: {
    name: "Modern & Engaging",
    description: "Contemporary design for engaging, interactive training",
    colors: {
      primary: "#059669", // Emerald
      secondary: "#64748b", // Slate
      accent: "#f472b6", // Pink
      background: "#ecfdf5",
      text: "#111827",
    },
    typography: {
      headingFont: "Poppins",
      bodyFont: "Nunito",
      style: "casual" as const,
    },
    writingTone: {
      tone: "engaging" as const,
      voice: "second-person" as const,
      complexity: "simple" as const,
    },
  },
};

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const DESIGNER_SYSTEM_PROMPT = `You are The Visual Designer - a specialized agent focused on defining the aesthetic and tone for online training courses.

## Your Role

You create cohesive visual and tonal identities that:

1. **Match the Industry** - Reflect industry norms and expectations
2. **Engage the Audience** - Appeal to the target learners
3. **Support Learning** - Enhance rather than distract from content
4. **Maintain Consistency** - Create a unified experience throughout
5. **Express Brand** - Incorporate any branding requirements

## Your Tools

### Visual Selection Tools (HITL - User selects from visual grids)
- **selectImageTypes** - Let user choose image types (Photos, Illustrations, 3D, Icons) - MULTI-SELECT
- **selectAllImageStyles** - COMBINED tool for style selection - presents multi-step wizard for ALL image types at once. Pass selected types as comma-separated string (e.g., "photo,illustration,3d")
- **selectColorPalette** - Let user choose a color scheme - SINGLE-SELECT
- **selectTypography** - Let user choose font pairing - SINGLE-SELECT
- **selectAnimationStyle** - Let user choose animation preferences - SINGLE-SELECT
- **selectComponentStyles** - Let user choose component variant preferences - MULTI-SELECT

### User Interaction
- **offerOptions** - Present text-based options for user to choose from
- **askClarifyingQuestions** - Ask questions to understand preferences

### Asset Discovery
- **searchMicroverse** - Find existing brand assets, logos, images in the media library
- **getMicroverseDetails** - Get details about specific assets

## Design Process (Phase-Based Workflow)

1. **Gather Preferences** - Brief discussion to understand basic needs
2. **Select Image Types** - User picks which image types to use (can select multiple)
3. **Select Image Styles** - Combined wizard for all selected types
4. **Select Colors** - User picks color palette
5. **Select Typography** - User picks font pairing
6. **Select Animations** - User picks animation style
7. **Select Components** - User picks component style preferences
8. **Finalize Design** - Compile all selections into final design spec

IMPORTANT: Each phase uses ONE tool call. Wait for user selection before transitioning to the next phase.

## Extended Design System

The design specification now includes:

### Image Styles (array - supports multiple types)
- Type: photo, illustration, 3d, icon
- Style: varies per type (corporate, flat, isometric, etc.)

### Color System (extended)
- Basic colors (primary, secondary, accent, background, text)
- Gradients (gradientPrimary, gradientSecondary)
- Surface colors (surfaceCard, surfaceElevated)
- Shadow intensity (none, subtle, medium, strong)

### Layout Preferences
- Container padding (compact, normal, spacious)
- Border radius (none, subtle, rounded, pill)
- Card style (flat, elevated, outlined, glass)

### Animation Settings
- Enabled (true/false)
- Style (subtle, moderate, dynamic)
- Duration (fast, normal, slow)
- Entrance type (fade, slide, scale)

### Component Preferences
- Header style (minimal, gradient, hero, split)
- Question style (standard, card, gamified)
- CTA style (solid, outline, gradient)

## Output Format

Create a complete design specification (JSON):

\`\`\`json
{
  "theme": "Theme Name",
  "colors": {
    "primary": "#hex",
    "secondary": "#hex",
    "accent": "#hex",
    "background": "#hex",
    "text": "#hex"
  },
  "typography": {
    "headingFont": "Font Name",
    "bodyFont": "Font Name",
    "style": "formal|casual|technical|friendly"
  },
  "writingTone": {
    "tone": "professional|conversational|academic|engaging",
    "voice": "first-person|second-person|third-person",
    "complexity": "simple|intermediate|advanced"
  },
  "imageStyles": [
    { "type": "photo", "style": "corporate", ... },
    { "type": "illustration", "style": "flat", ... }
  ],
  "colorSystem": {
    "gradientPrimary": "from-blue-500 to-blue-700",
    "surfaceCard": "#ffffff",
    "shadowIntensity": "subtle"
  },
  "layout": {
    "containerPadding": "normal",
    "borderRadius": "rounded",
    "cardStyle": "elevated"
  },
  "animation": {
    "enabled": true,
    "style": "moderate",
    "duration": "normal",
    "entranceType": "slide"
  },
  "componentPreferences": {
    "headerStyle": "hero",
    "questionStyle": "card",
    "ctaStyle": "solid"
  },
  "branding": { ... },
  "notes": "..."
}
\`\`\`

## Guidelines

- Use visual selection tools to gather user preferences - they see thumbnail grids
- The visual tools will return structured config data from user selections
- Compile all selection responses into the final design specification
- Consider accessibility (color contrast, readability)
- Match formality to the subject matter

Remember: Let users SEE their options through visual selectors, not just text descriptions.`;

// ============================================================================
// VISUAL DESIGNER NODE FUNCTION
// ============================================================================

/**
 * The Visual Designer agent node.
 * Defines course aesthetics through user interaction.
 */
export async function visualDesignerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[visual_designer] ============ Visual Designer Agent ============");
  console.log("  Project brief available:", state.projectBrief ? "yes" : "no");
  console.log("  Existing design:", state.visualDesign ? "yes" : "no");

  // Determine current phase from awaitingUserAction state
  const workState = state.awaitingUserAction;
  const currentPhase: VisualDesignerPhase = (workState?.agent === "visual_designer" && workState?.phase) 
    ? (workState.phase as VisualDesignerPhase)
    : "gathering_preferences";  // Default to first phase
  
  const phaseConfig = VISUAL_DESIGNER_PHASES[currentPhase];
  console.log(`  Current phase: ${currentPhase} - ${phaseConfig.description}`);
  console.log(`  Allowed tools: ${phaseConfig.allowedTools.join(", ") || "none"}`);

  // Get frontend tools from CopilotKit state
  // PHASE-GATING: Only allow tools permitted in the current phase
  const frontendActions = state.copilotkit?.actions ?? [];
  const allDesignerToolNames = [
    // User interaction tools
    "offerOptions",
    "askClarifyingQuestions",
    // Visual selection tools (HITL)
    "selectImageTypes",
    "selectAllImageStyles",
    "selectColorPalette",
    "selectTypography",
    "selectAnimationStyle",
    "selectComponentStyles",
    // Asset discovery tools
    "searchMicroverse",
    "getMicroverseDetails",
  ];
  
  // Filter to only tools allowed in current phase
  const frontendDesignerTools = frontendActions.filter((action: { name: string }) =>
    allDesignerToolNames.includes(action.name) && phaseConfig.allowedTools.includes(action.name)
  );

  // Combine frontend tools with backend web_search tool (if in appropriate phase)
  const designerTools = webSearchTool && phaseConfig.allowedTools.includes("web_search")
    ? [...frontendDesignerTools, webSearchTool]
    : frontendDesignerTools;

  console.log("  Available tools (phase-filtered):", designerTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = DESIGNER_SYSTEM_PROMPT;
  
  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }
  
  // Add phase-specific instructions
  systemContent += `\n\n## CURRENT PHASE: ${currentPhase.toUpperCase()}

**You are in the "${currentPhase}" phase.** ${phaseConfig.description}

ALLOWED TOOLS IN THIS PHASE: ${phaseConfig.allowedTools.join(", ") || "NONE - output your final design spec"}

${currentPhase === "gathering_preferences" ? `
### Phase Instructions
1. Briefly greet the user and explain you'll help define the visual design
2. Ask about any existing brand guidelines or preferences using askClarifyingQuestions
3. You may search Microverse for existing brand assets
4. When ready to proceed, output: [PHASE: selecting_image_types]
` : ""}
${currentPhase === "selecting_image_types" ? `
### Phase Instructions
1. Call selectImageTypes to show the user image type options
2. Users can select MULTIPLE types (e.g., Photos AND Illustrations)
3. After user confirms selection, output: [PHASE: selecting_image_styles]
` : ""}
${currentPhase === "selecting_image_styles" ? `
### Phase Instructions
1. Call selectAllImageStyles ONCE with ALL the image types the user selected
2. Pass the types as a comma-separated string: selectAllImageStyles({ imageTypes: "photo,illustration,3d" })
3. DO NOT call this tool multiple times - it handles all types in one multi-step wizard
4. The user will step through each type in the UI and select a style for each
5. After the user confirms all selections, output: [PHASE: selecting_colors]
` : ""}
${currentPhase === "selecting_colors" ? `
### Phase Instructions
1. Call selectColorPalette ONCE to let user choose a color scheme
2. Wait for user selection
3. After user confirms, output: [PHASE: selecting_typography]
` : ""}
${currentPhase === "selecting_typography" ? `
### Phase Instructions
1. Call selectTypography ONCE to let user choose font pairing
2. Wait for user selection
3. After user confirms, output: [PHASE: selecting_animations]
` : ""}
${currentPhase === "selecting_animations" ? `
### Phase Instructions
1. Call selectAnimationStyle ONCE to let user choose animation preferences
2. Wait for user selection
3. After user confirms, output: [PHASE: selecting_components]
` : ""}
${currentPhase === "selecting_components" ? `
### Phase Instructions
1. Call selectComponentStyles ONCE to let user choose component variant preferences
2. Wait for user selection
3. After user confirms, output: [PHASE: finalizing]
` : ""}
${currentPhase === "finalizing" ? `
### Phase Instructions
1. NO TOOL CALLS - compile all selections into final design specification
2. Use the config data returned from each selection tool
3. Output the complete design as a JSON code block
4. End with [DESIGN COMPLETE]
` : ""}`;

  // Include project brief
  if (state.projectBrief) {
    const condensedBrief = getCondensedBrief(state.projectBrief);
    systemContent += `\n\n## Project Context\n\n${condensedBrief}`;

    // Suggest themes based on industry
    const industry = state.projectBrief.industry.toLowerCase();
    systemContent += `\n\n## Recommended Themes for "${state.projectBrief.industry}"

Based on the industry context, consider these pre-defined themes:`;

    // Match industry to themes
    if (industry.includes("health") || industry.includes("medical") || industry.includes("safety")) {
      systemContent += "\n- Healthcare & Safety - Trust-inspiring for medical training";
    }
    if (industry.includes("tech") || industry.includes("engineer") || industry.includes("it")) {
      systemContent += "\n- Technical & Engineering - Modern tech-focused design";
    }
    if (industry.includes("hotel") || industry.includes("hospitality") || industry.includes("service")) {
      systemContent += "\n- Hospitality & Service - Warm, welcoming design";
    }
    if (industry.includes("education") || industry.includes("academic") || industry.includes("university")) {
      systemContent += "\n- Education & Academic - Classic academic style";
    }
    systemContent += "\n- Corporate Professional - Clean business design";
    systemContent += "\n- Modern & Engaging - Contemporary interactive design";
  }

  // Include existing design if refining
  if (state.visualDesign) {
    systemContent += `\n\n## Current Design (to refine)

Theme: ${state.visualDesign.theme}
Primary Color: ${state.visualDesign.colors.primary}
Tone: ${state.visualDesign.writingTone.tone}

The user may want to adjust specific elements of this design.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = designerTools.length > 0
    ? designerModel.bindTools(designerTools)
    : designerModel;

  // Filter messages for this agent's context - filter orphans first, then slice
  // Filter AFTER slicing - slicing can create new orphans by removing AI messages with tool_use
  const slicedMessages = (state.messages || []).slice(-8);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[visual_designer]");

  console.log("  Invoking visual designer model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Visual designer response received");

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
1. Greet the user briefly
2. Use the offerOptions tool to present design theme choices
3. OR ask about their design preferences

The user is waiting for design options.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...recentMessages, nudgeMessage],
      config
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

  // Check for design completion markers
  const isDesignComplete = responseText.toLowerCase().includes("[design complete]") ||
    responseText.toLowerCase().includes("[done]") ||
    responseText.toLowerCase().includes("visual design:");
  
  // Parse visual design on completion
  let parsedDesign: VisualDesign | null = null;
  if (isDesignComplete) {
    console.log("  [visual_designer] Design completion detected - parsing visual design");
    parsedDesign = parseVisualDesign(responseText);
    if (parsedDesign) {
      console.log("  [visual_designer] Parsed visual design:", {
        theme: parsedDesign.theme,
        tone: parsedDesign.writingTone.tone,
        style: parsedDesign.typography.style,
      });
    } else {
      console.log("  [visual_designer] WARNING: Could not parse structured visual design from response");
    }
  }

  // Check for phase transition markers
  const phaseTransitionMatch = responseText.match(/\[PHASE:\s*(\w+)\]/i);
  let nextPhase: VisualDesignerPhase | null = null;
  if (phaseTransitionMatch) {
    const requestedPhase = phaseTransitionMatch[1].toLowerCase() as VisualDesignerPhase;
    if (requestedPhase in VISUAL_DESIGNER_PHASES) {
      nextPhase = requestedPhase;
      console.log(`  [visual_designer] Phase transition detected: ${currentPhase} -> ${nextPhase}`);
    }
  }

  // Check if response has HITL tool calls that require user interaction
  const HITL_TOOLS = ["offerOptions", "askClarifyingQuestions"];
  const hasHITLToolCall = aiResponse.tool_calls?.some(tc => 
    HITL_TOOLS.includes(tc.name)
  );
  
  // Check if any tool calls were made (HITL or otherwise)
  const hasAnyToolCall = aiResponse.tool_calls && aiResponse.tool_calls.length > 0;
  
  // Determine the new work state
  let newWorkState: AgentWorkState | null = null;
  
  if (isDesignComplete) {
    // Work complete - clear the state
    console.log("  [visual_designer] Work complete - clearing awaitingUserAction");
    newWorkState = null;
  } else if (nextPhase) {
    // Phase transition - update to new phase
    const newPhaseConfig = VISUAL_DESIGNER_PHASES[nextPhase];
    console.log(`  [visual_designer] Entering phase: ${nextPhase}`);
    newWorkState = {
      agent: "visual_designer",
      phase: nextPhase,
      allowedTools: [...newPhaseConfig.allowedTools],
    };
  } else if (hasHITLToolCall) {
    // HITL tool call - stay in current phase, mark pending tool
    const hitlTool = aiResponse.tool_calls?.find(tc => HITL_TOOLS.includes(tc.name));
    console.log(`  [visual_designer] HITL tool call (${hitlTool?.name}) - awaiting user response`);
    newWorkState = {
      agent: "visual_designer",
      phase: currentPhase,
      pendingTool: hitlTool?.name,
      allowedTools: [...phaseConfig.allowedTools],
    };
  } else if (hasAnyToolCall) {
    // Non-HITL tool call - stay in current phase, continue working
    console.log(`  [visual_designer] Tool calls made - continuing in phase ${currentPhase}`);
    newWorkState = {
      agent: "visual_designer",
      phase: currentPhase,
      allowedTools: [...phaseConfig.allowedTools],
    };
  } else {
    // No tool calls, no phase transition - keep working
    console.log(`  [visual_designer] No tool calls - continuing in phase ${currentPhase}`);
    newWorkState = {
      agent: "visual_designer",
      phase: currentPhase,
      allowedTools: [...phaseConfig.allowedTools],
    };
  }

  // Build progress update for activeTask
  const progressUpdates: string[] = [];
  if (nextPhase === "presenting_options") {
    progressUpdates.push("Visual Designer: Completed design research phase");
  }
  if (nextPhase === "creating_spec") {
    progressUpdates.push("Visual Designer: User selected design options");
  }
  if (isDesignComplete && parsedDesign) {
    progressUpdates.push(`Visual Designer: Created design spec with theme "${parsedDesign.theme}"`);
  }

  // Update activeTask with progress if there are updates
  const activeTaskUpdate: Partial<ActiveTask> | null = progressUpdates.length > 0
    ? {
        progress: progressUpdates,
        ...(isDesignComplete && { assignedAgent: "orchestrator" as const }),
      }
    : null;

  return {
    messages: [response],
    currentAgent: "visual_designer",
    agentHistory: ["visual_designer"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
    // Include parsed visual design if available
    ...(parsedDesign && { visualDesign: parsedDesign }),
    // Update work state based on phase/HITL analysis
    awaitingUserAction: newWorkState,
    // Update activeTask with progress (reducer will merge with existing progress)
    ...(activeTaskUpdate && { activeTask: activeTaskUpdate as ActiveTask }),
  };
}

/**
 * Parses a visual designer's text response to extract design specification.
 */
export function parseVisualDesign(content: string): VisualDesign | null {
  try {
    // Look for JSON block
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      return validateVisualDesign(parsed);
    }
    return null;
  } catch (error) {
    console.error("[visual_designer] Failed to parse design:", error);
    return null;
  }
}

function validateVisualDesign(input: Partial<VisualDesign>): VisualDesign {
  return {
    theme: input.theme || "Custom",
    colors: {
      primary: input.colors?.primary || "#1e40af",
      secondary: input.colors?.secondary || "#64748b",
      accent: input.colors?.accent || "#0ea5e9",
      background: input.colors?.background || "#f8fafc",
      text: input.colors?.text || "#1e293b",
    },
    typography: {
      headingFont: input.typography?.headingFont || "Inter",
      bodyFont: input.typography?.bodyFont || "Open Sans",
      style: input.typography?.style || "formal",
    },
    writingTone: {
      tone: input.writingTone?.tone || "professional",
      voice: input.writingTone?.voice || "second-person",
      complexity: input.writingTone?.complexity || "intermediate",
    },
    branding: input.branding,
    notes: input.notes,
  };
}

/**
 * Gets a theme preset by name.
 */
export function getThemePreset(themeName: string): Partial<VisualDesign> | null {
  const key = Object.keys(DESIGN_THEMES).find(
    (k) => k.toLowerCase() === themeName.toLowerCase() ||
           DESIGN_THEMES[k as keyof typeof DESIGN_THEMES].name.toLowerCase().includes(themeName.toLowerCase())
  );

  if (key) {
    const theme = DESIGN_THEMES[key as keyof typeof DESIGN_THEMES];
    return {
      theme: theme.name,
      colors: theme.colors,
      typography: theme.typography,
      writingTone: theme.writingTone,
    };
  }

  return null;
}

/**
 * Suggests appropriate themes based on industry context.
 */
export function suggestThemes(industry: string): string[] {
  const suggestions: string[] = [];
  const lower = industry.toLowerCase();

  if (lower.includes("health") || lower.includes("medical") || lower.includes("safety")) {
    suggestions.push("healthcare");
  }
  if (lower.includes("tech") || lower.includes("engineer") || lower.includes("it") || lower.includes("software")) {
    suggestions.push("technical");
  }
  if (lower.includes("hotel") || lower.includes("hospitality") || lower.includes("service") || lower.includes("retail")) {
    suggestions.push("hospitality");
  }
  if (lower.includes("education") || lower.includes("academic") || lower.includes("university") || lower.includes("school")) {
    suggestions.push("education");
  }

  // Always include corporate and modern as options
  suggestions.push("corporate", "modern");

  return [...new Set(suggestions)];
}

export default visualDesignerNode;

