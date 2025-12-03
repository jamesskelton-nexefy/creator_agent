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
import type { OrchestratorState, VisualDesign } from "../state/agent-state";
import { getCondensedBrief } from "../state/agent-state";
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

### User Interaction
- **offerOptions** - Present 2-3 design options for user to choose from

### Research
- **web_search** - Research design trends, industry standards, color psychology, accessibility best practices

### Asset Discovery
- **searchMicroverse** - Find existing brand assets, logos, images in the media library
- **getMicroverseDetails** - Get details about specific assets

## Design Process

1. **Analyze Context** - Review the project brief and audience
2. **Propose Themes** - Offer 2-3 appropriate design directions
3. **Refine Selection** - Get user preference on key elements
4. **Document Design** - Create comprehensive design specification

## Design Elements

### Color Palette
- **Primary** - Main brand/accent color
- **Secondary** - Supporting color
- **Accent** - Highlights and CTAs
- **Background** - Page backgrounds
- **Text** - Body text color

### Typography
- **Heading Font** - For titles and headers
- **Body Font** - For content text
- **Style** - formal, casual, technical, or friendly

### Writing Tone
- **Tone** - professional, conversational, academic, or engaging
- **Voice** - first-person, second-person, or third-person
- **Complexity** - simple, intermediate, or advanced

## Output Format

Create a complete design specification:

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
  "branding": {
    "companyName": "If provided",
    "brandGuidelines": "Any specific requirements"
  },
  "notes": "Additional style notes"
}
\`\`\`

## Guidelines

- Consider accessibility (color contrast, readability)
- Match formality to the subject matter
- Account for industry conventions
- Balance aesthetics with functionality
- Keep design decisions practical and implementable

Remember: Good design is invisible - it supports the content without getting in the way.`;

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

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  const frontendDesignerTools = frontendActions.filter((action: { name: string }) =>
    [
      // User interaction
      "offerOptions",
      // Media/asset search
      "searchMicroverse",
      "getMicroverseDetails",
    ].includes(action.name)
  );

  // Combine frontend tools with backend web_search tool
  const designerTools = webSearchTool
    ? [...frontendDesignerTools, webSearchTool]
    : frontendDesignerTools;

  console.log("  Available tools:", designerTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = DESIGNER_SYSTEM_PROMPT;

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

  return {
    messages: [response],
    currentAgent: "visual_designer",
    agentHistory: ["visual_designer"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
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

