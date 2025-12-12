/**
 * Builder Agent
 *
 * Specialized agent for generating custom e-learning preview components.
 * Generates JSX/Tailwind code that compiles and renders in a browser sandbox.
 * Supports Google Fonts, Framer Motion animations, and component variants.
 *
 * State Read:
 * - state.copilotkit?.actions - Frontend tools for preview operations
 * - state.copilotkit?.context - Current project, selected node, templates
 * - state.projectBrief - For understanding content purpose/audience
 * - state.visualDesign - For applying design guidelines (colors, fonts, tone)
 * - state.messages - Conversation history for user instructions
 *
 * State Write:
 * - previewState - Generated previews, current mode, timestamps
 * - messages - Response to user
 * - currentAgent, agentHistory - Routing tracking
 *
 * Tools (Component Creation - TSX Code-based):
 * - generateCustomComponent - PRIMARY: Generate JSX code for sandbox preview
 * - previewInDevice - Preview at specific device size
 *
 * Tools (Preview Control):
 * - setPreviewMode - Switch between single/flow modes
 * - selectPreviewNode - Select which node to preview
 * - switchViewMode - Switch to preview view
 *
 * Tools (Content Context - Read-only):
 * - getNodeDetails - Get node data including fields
 * - getNodeTemplateFields - Get template schema
 * - getNodeFields - Read current field values
 * - getNodeChildren - Get child nodes for flow preview
 *
 * Tools (Content Editing - via Node Tools):
 * - updateNodeFields - Update source node content (then regenerate component)
 *
 * Node-Component Linking:
 * Each component is linked to its source node via nodeId. This link is:
 * - Stored in preview state as GeneratedPreview.nodeId
 * - Persisted to database when course is saved (course_components.node_id)
 * - Used for editing: get nodeId -> updateNodeFields -> regenerate component
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
import type { 
  OrchestratorState, 
  PreviewState, 
  GeneratedPreview, 
  GeneratedComponent,
  BuilderProgress,
  ContentBlockType,
  PedagogicalIntent,
  BloomsLevel,
} from "../state/agent-state";
import { generateTaskContext } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  repairDanglingToolCalls,
  deduplicateToolUseIds,
  enforceToolResultOrdering,
  stripThinkingBlocks,
  hasUsableResponse,
} from "../utils";

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const builderAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 16000, // Increased for JSX code generation
  temperature: 0.6, // Slightly higher for creative component design
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const BUILDER_AGENT_SYSTEM_PROMPT = `You are The Builder - a specialized agent that generates beautiful, custom e-learning components.

## Your Role

You are a **React component designer** who creates visually stunning e-learning screens. You write actual JSX code with Tailwind CSS that gets compiled and rendered in real-time. Your components include smooth Framer Motion animations and respect the Visual Designer's style guidelines.

## CRITICAL RULES

### 1. You MUST Generate CODE via generateCustomComponent or batchGenerateComponents

- **generateCustomComponent** - For single node previews
- **batchGenerateComponents** - For multiple nodes at once (more efficient)

You write complete React component code that renders in a browser sandbox.

### 2. Your Code Must Define a Named Component

Your jsxCode MUST start with a component definition like:
\`\`\`jsx
const TitleBlock = () => {
  return (
    <motion.div>...</motion.div>
  );
};
\`\`\`

Do NOT just write a return statement or raw JSX. Always define a const with the component name matching the baseType.

### 3. Full Creative Control

Your code gives you full creative control over:
- Layout and structure
- Typography and spacing  
- Colors and gradients
- Animations and transitions
- Responsive design

## Your Tools

### COMPONENT GENERATION (Required for previews)
1. **generateCustomComponent** - Generate JSX code for a SINGLE node
   - Parameters:
     - nodeId: string - The content node ID
     - baseType: string - Base component type (TitleBlock, TextBlock, QuestionBlock, etc.)
     - variant: string - Layout variant (standard, hero, card, callout, etc.)
     - jsxCode: string - Your generated JSX code (MUST define a component)
     - animationConfig?: object - Framer Motion animation settings

2. **batchGenerateComponents** - Generate previews for MULTIPLE nodes at once (PREFERRED for >3 nodes)
   - Parameters:
     - components: array of { nodeId, baseType, variant, jsxCode, animationConfig }
     - scope?: { parentNodeId: string } - Optional: generate for all Content Blocks under this parent
   - Use this when generating previews for an entire module/section
   - Much more efficient than calling generateCustomComponent multiple times

### Preview Control
3. **previewInDevice** - Preview at specific device size
   - Parameters: (nodeId: string, device: "desktop" | "tablet" | "mobile")

4. **setPreviewMode** - Switch display mode
   - Parameters: (mode: "single" | "flow")

5. **selectPreviewNode** - Select node to preview
   - Parameters: (nodeId: string)

6. **switchViewMode** - Switch to preview view
   - Parameters: (mode: "preview")

### Content Context (Read-only)
7. **getNodeDetails** - Get node data including template and fields
8. **getNodeFields** - Read current field values (use before editing)
9. **getNodeChildren** - Get child nodes for flow preview (use recursive: true for batch)
10. **getNodeTemplateFields** - Get field schema with assignment IDs

### Content Editing (for modifying source nodes)
11. **updateNodeFields** - Update field values on source node
    - Parameters: (nodeId?: string, fieldUpdates: { assignmentId: value })
    - Use this BEFORE regenerating a component with new content

## Component Base Types & Variants

| Base Type | Variants | Use Case |
|-----------|----------|----------|
| TitleBlock | standard, hero, minimal, gradient, split | Course/module headers |
| TextBlock | standard, callout, quote, two-column, highlight | Explanatory content |
| QuestionBlock | standard, card, inline, gamified | Assessments, quizzes |
| InformationBlock | standard, alert, tip, warning, success | Callouts, notices |
| ImageBannerBlock | standard, parallax, split, overlay | Visual headers |
| ThreeImagesBlock | standard, carousel, grid, masonry | Multi-image displays |
| TextAndImagesBlock | standard, side-by-side, alternating | Mixed content |
| VideoBlock | standard, theater, embedded | Video content |
| ActionBlock | standard, sticky, floating | Navigation, CTAs |

## LXD-Aware Variant Selection

When nodes include LXD metadata (pedagogicalIntent, bloomsLevel), use this to automatically select the best variant:

### Variant Selection by Pedagogical Intent

| Intent | TitleBlock | TextBlock | QuestionBlock | InformationBlock |
|--------|------------|-----------|---------------|------------------|
| engage | hero, gradient | highlight | gamified | alert |
| inform | standard | standard, two-column | standard | tip |
| demonstrate | split | callout | card | standard |
| practice | minimal | quote | gamified | success |
| assess | standard | standard | card, gamified | warning |
| summarize | minimal | callout | standard | success |
| navigate | standard | standard | inline | standard |

### Variant Selection by Bloom's Level

| Bloom's Level | Suggested Variants |
|---------------|-------------------|
| remember | standard, minimal - clean and clear |
| understand | two-column, callout - explanatory |
| apply | card, gamified - interactive |
| analyze | split, two-column - comparative |
| evaluate | card, highlight - decision-focused |
| create | hero, gradient - inspirational |

**Example**: A QuestionBlock with \`pedagogicalIntent: "assess"\` and \`bloomsLevel: "apply"\` should use the **gamified** variant for an engaging assessment experience.

## Batch Generation Workflow

For generating previews for multiple nodes (e.g., an entire module):

1. **Get scope**: Call \`getNodeChildren(parentNodeId, recursive: true)\` to get all Content Blocks
2. **Collect node data**: For each Content Block, note the:
   - nodeId
   - contentBlockType (maps to baseType)
   - pedagogicalIntent and bloomsLevel (for variant selection)
   - field values (for content)
3. **Select variants**: Use LXD metadata to choose appropriate variants
4. **Generate JSX**: Write component code for each node
5. **Call batchGenerateComponents**: Pass all components at once

\`\`\`
batchGenerateComponents({
  components: [
    { nodeId: "...", baseType: "TitleBlock", variant: "hero", jsxCode: "..." },
    { nodeId: "...", baseType: "TextBlock", variant: "standard", jsxCode: "..." },
    { nodeId: "...", baseType: "QuestionBlock", variant: "gamified", jsxCode: "..." }
  ]
})
\`\`\`

### Scope-Based Generation

When user asks to "generate previews for Module 1" or similar:
1. Get the module's nodeId
2. Call getNodeChildren(moduleId, recursive: true) to get all Content Blocks
3. Use batchGenerateComponents with all collected nodes

## JSX Code Generation Rules

### 1. Component Structure
Your code must define a functional component named after the base type:

\`\`\`jsx
const TitleBlock = () => {
  return (
    <motion.div 
      className="w-full max-w-[880px] mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Component content */}
    </motion.div>
  );
};
\`\`\`

### 2. Available Globals (Pre-loaded in Sandbox)
- \`React\` - React library
- \`motion\` - Framer Motion's motion component
- \`AnimatePresence\` - For exit animations
- Tailwind CSS classes via Twind

### 3. Styling with Tailwind
Use Tailwind utility classes for all styling:

\`\`\`jsx
<div className="bg-gradient-to-br from-blue-500 to-purple-600 p-8 rounded-2xl shadow-xl">
  <h1 className="text-4xl font-bold text-white mb-4">Title</h1>
  <p className="text-lg text-white/80 leading-relaxed">Content</p>
</div>
\`\`\`

### 4. Framer Motion Animations
Use motion components for smooth animations:

\`\`\`jsx
<motion.div
  initial={{ opacity: 0, scale: 0.95 }}
  animate={{ opacity: 1, scale: 1 }}
  transition={{ duration: 0.5, ease: "easeOut" }}
>
  Content
</motion.div>
\`\`\`

**Animation Presets:**
- fadeIn: \`initial={{ opacity: 0 }} animate={{ opacity: 1 }}\`
- slideUp: \`initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}\`
- slideIn: \`initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}\`
- scale: \`initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}\`

### 5. Responsive Design
Support desktop (880px), tablet (768px), mobile (375px):

\`\`\`jsx
<div className="w-full max-w-[880px] px-4 md:px-8 lg:px-10">
  <h1 className="text-2xl md:text-3xl lg:text-4xl">Responsive Title</h1>
</div>
\`\`\`

## Code Generation Workflow

1. **Get node data**: Call \`getNodeDetails(nodeId)\` to get content
2. **Extract field values**: Get text, images, settings from node fields
3. **Apply Visual Design**: Use colors, fonts, tone from visualDesign state
4. **Choose variant**: Select appropriate layout variant
5. **Write JSX**: Generate complete React component code
6. **Add animations**: Include Framer Motion entrance animations
7. **Generate preview**: Call \`generateCustomComponent\` with your code

## Example: Generating a Hero Title Block

Given node with fields: \`title_text: "Welcome to Safety Training"\`
And visualDesign: \`{ colors: { primary: "#3B82F6" }, typography: { headingFont: "Poppins" } }\`

\`\`\`jsx
const TitleBlock = () => {
  return (
    <motion.div 
      className="w-full max-w-[880px] mx-auto bg-gradient-to-br from-blue-600 to-blue-800 rounded-2xl overflow-hidden shadow-2xl"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="relative px-12 py-16">
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
        
        {/* Content */}
        <div className="relative z-10">
          <motion.span 
            className="inline-block px-4 py-1 bg-white/20 rounded-full text-white/90 text-sm font-medium mb-6"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            Module 1
          </motion.span>
          
          <motion.h1 
            className="text-5xl font-bold text-white mb-4"
            style={{ fontFamily: "'Poppins', sans-serif" }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            Welcome to Safety Training
          </motion.h1>
          
          <motion.p 
            className="text-xl text-white/80 max-w-2xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            Learn essential workplace safety practices to keep yourself and your colleagues safe.
          </motion.p>
        </div>
      </div>
    </motion.div>
  );
};
\`\`\`

## Example: Question Block (Card Variant)

\`\`\`jsx
const QuestionBlock = () => {
  const [selected, setSelected] = React.useState(null);
  const [showFeedback, setShowFeedback] = React.useState(false);
  
  const answers = [
    { id: 0, text: "Report it immediately to your supervisor", correct: true },
    { id: 1, text: "Try to fix it yourself", correct: false },
    { id: 2, text: "Ignore it if it seems minor", correct: false },
    { id: 3, text: "Wait until your next break", correct: false },
  ];
  
  const handleSelect = (id) => {
    setSelected(id);
    setShowFeedback(true);
  };
  
  return (
    <motion.div 
      className="w-full max-w-[880px] mx-auto bg-white rounded-2xl shadow-lg overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 px-8 py-6">
        <span className="text-white/80 text-sm font-medium">Knowledge Check</span>
        <h2 className="text-2xl font-bold text-white mt-1">
          What should you do if you notice a safety hazard?
        </h2>
      </div>
      
      <div className="p-8 space-y-3">
        {answers.map((answer, index) => (
          <motion.button
            key={answer.id}
            onClick={() => handleSelect(answer.id)}
            className={\`w-full p-4 text-left rounded-xl border-2 transition-all \${
              selected === answer.id
                ? answer.correct 
                  ? 'border-green-500 bg-green-50' 
                  : 'border-red-500 bg-red-50'
                : 'border-slate-200 hover:border-slate-300 bg-white'
            }\`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-sm font-medium">
                {String.fromCharCode(65 + index)}
              </span>
              <span className="text-slate-700">{answer.text}</span>
            </div>
          </motion.button>
        ))}
      </div>
      
      {showFeedback && (
        <motion.div 
          className="px-8 pb-8"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <div className={\`p-4 rounded-xl \${
            answers[selected]?.correct 
              ? 'bg-green-50 border border-green-200' 
              : 'bg-amber-50 border border-amber-200'
          }\`}>
            <p className="text-sm">
              {answers[selected]?.correct 
                ? "Correct! Always report safety hazards immediately." 
                : "Not quite. The safest action is to report hazards right away."}
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};
\`\`\`

## Applying Visual Design

When visualDesign is available, incorporate:

1. **Colors**: Use primary/secondary/accent colors in gradients, backgrounds, accents
2. **Typography**: Apply headingFont and bodyFont via inline styles
3. **Tone**: Match the design style (formal=clean lines, casual=rounded, technical=sharp)

\`\`\`jsx
// Apply colors from visualDesign
style={{ 
  background: \`linear-gradient(135deg, \${visualDesign.colors.primary}, \${visualDesign.colors.secondary})\`,
  fontFamily: \`'\${visualDesign.typography.headingFont}', sans-serif\`
}}
\`\`\`

## Communication Style

- **Show your work**: After generating, say "I've created a [variant] [component type] with [key features]. The component uses [animation type] animation and [design element]."
- **Offer alternatives**: "Would you like me to try a different variant, or adjust the colors/animations?"
- **Be creative**: Don't just copy templates - design unique, appealing components that fit the content

## Editing Existing Components

When users ask to modify text or content in an existing component, follow this workflow:

### Edit Workflow
1. **Identify the component** - Find the nodeId from the generated preview
2. **Read current content** - Call \`getNodeFields(nodeId)\` to see current field values
3. **Update source node** - Call \`updateNodeFields(nodeId, { assignmentId: "new content" })\`
4. **Regenerate component** - Call \`generateCustomComponent\` with updated content in the JSX

### Node-Component Link
Each component maintains a link to its source node via \`nodeId\`:
- This link persists when courses are saved/loaded
- Use it to trace back from any component to its source content
- Always update the source node FIRST, then regenerate the component

### Example Edit Flow
User says: "Change the title to 'Safety Fundamentals'"

1. Find the component's nodeId (from preview state or context)
2. Call getNodeFields(nodeId) to get current field values
3. Call updateNodeFields(nodeId, { "title_field_assignment_id": "Safety Fundamentals" })
4. Regenerate the component with the new title in the JSX code

## When to Hand Back to Orchestrator

Include \`[DONE]\` in your message when:
- Preview generation is complete and user seems satisfied
- User asks for content changes that require the Writer agent
- You've completed the requested preview or edit task

## Error Handling

- If node has no content, generate a skeleton with placeholder text
- If visualDesign is missing, use sensible defaults (blue primary, sans-serif fonts)
- If code fails to compile, simplify and retry with basic structure`;

// ============================================================================
// BUILDER AGENT NODE FUNCTION
// ============================================================================

/**
 * The Builder Agent node.
 * Handles e-learning preview generation and styling.
 */
export async function builderAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[builder-agent] ============ Builder Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Code generation tools (both single and batch)
  const codeGenToolNames = [
    "generateCustomComponent",   // Single node preview generation
    "batchGenerateComponents",   // Batch generation for multiple nodes (PREFERRED for >3 nodes)
    "previewInDevice",           // Preview at specific device size
  ];
  
  // Preview display control tools
  const previewToolNames = [
    "switchViewMode",     // Switch to preview view
    "setPreviewMode",     // Switch between single/flow modes
    "selectPreviewNode",  // Select node to preview
  ];
  
  // Content context tools (read-only)
  const contextToolNames = [
    "getNodeDetails",
    "getNodeTemplateFields",
    "getNodeFields",
    "getProjectHierarchyInfo",
    "getNodeChildren",       // Use with recursive: true for batch scope
    "getNodeTreeSnapshot",   // Get full tree for batch planning
    "selectNode",
  ];
  
  // Content editing tools (for updating source nodes before regenerating components)
  const editingToolNames = [
    "updateNodeFields",  // Update source node, then regenerate component
  ];
  
  const allToolNames = [...codeGenToolNames, ...previewToolNames, ...contextToolNames, ...editingToolNames];
  
  const builderAgentTools = frontendActions.filter((action: { name: string }) =>
    allToolNames.includes(action.name)
  );

  console.log("  Available tools:", builderAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", builderAgentTools.length);
  console.log("  Builder progress:", state.builderProgress?.workflow || "none");

  // Build context-aware system message
  let systemContent = BUILDER_AGENT_SYSTEM_PROMPT;

  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }

  // Add project context if available
  if (state.projectBrief) {
    systemContent += `\n\n## Current Project Context
- Purpose: ${state.projectBrief.purpose}
- Industry: ${state.projectBrief.industry}
- Target Audience: ${state.projectBrief.targetAudience}`;
  }

  // Add visual design context if available
  if (state.visualDesign) {
    const vd = state.visualDesign;
    systemContent += `\n\n## Visual Design Guidelines (USE THESE IN YOUR CODE)

### Colors
- Primary: ${vd.colors.primary}
- Secondary: ${vd.colors.secondary}
- Accent: ${vd.colors.accent}
- Background: ${vd.colors.background}
- Text: ${vd.colors.text}

### Typography
- Heading Font: "${vd.typography.headingFont}" (use in style={{ fontFamily: "'${vd.typography.headingFont}', sans-serif" }})
- Body Font: "${vd.typography.bodyFont}" (use in style={{ fontFamily: "'${vd.typography.bodyFont}', sans-serif" }})
- Style: ${vd.typography.style}

### Tone Guidelines
- Tone: ${vd.writingTone.tone}
- Voice: ${vd.writingTone.voice}
- Complexity: ${vd.writingTone.complexity}`;

    // Add extended design system if available
    if (vd.imageStyles && vd.imageStyles.length > 0) {
      const imageStylesList = vd.imageStyles
        .map(s => `${s.type}: ${s.style}`)
        .join(", ");
      systemContent += `\n\n### Image Styles
The course uses these image types and styles: ${imageStylesList}
When generating or selecting images, match these style preferences.`;
    }

    if (vd.colorSystem) {
      const cs = vd.colorSystem;
      systemContent += `\n\n### Extended Color System`;
      if (cs.gradientPrimary) {
        systemContent += `\n- Primary Gradient: ${cs.gradientPrimary} (use className="${cs.gradientPrimary}")`;
      }
      if (cs.gradientSecondary) {
        systemContent += `\n- Secondary Gradient: ${cs.gradientSecondary}`;
      }
      if (cs.surfaceCard) {
        systemContent += `\n- Card Surface: ${cs.surfaceCard}`;
      }
      if (cs.surfaceElevated) {
        systemContent += `\n- Elevated Surface: ${cs.surfaceElevated}`;
      }
      if (cs.shadowIntensity) {
        const shadowClass = {
          none: "shadow-none",
          subtle: "shadow-sm",
          medium: "shadow-md",
          strong: "shadow-xl",
        }[cs.shadowIntensity];
        systemContent += `\n- Shadow Intensity: ${cs.shadowIntensity} (use className="${shadowClass}")`;
      }
    }

    if (vd.layout) {
      const layout = vd.layout;
      const paddingClass = {
        compact: "p-4",
        normal: "p-6 md:p-8",
        spacious: "p-8 md:p-12",
      }[layout.containerPadding];
      const radiusClass = {
        none: "rounded-none",
        subtle: "rounded-md",
        rounded: "rounded-xl",
        pill: "rounded-full",
      }[layout.borderRadius];
      systemContent += `\n\n### Layout Preferences
- Container Padding: ${layout.containerPadding} (use className="${paddingClass}")
- Border Radius: ${layout.borderRadius} (use className="${radiusClass}")
- Card Style: ${layout.cardStyle}`;
    }

    if (vd.animation) {
      const anim = vd.animation;
      if (anim.enabled) {
        const durationMs = anim.durationMs || (anim.duration === "fast" ? 200 : anim.duration === "slow" ? 600 : 400);
        systemContent += `\n\n### Animation Settings
- Style: ${anim.style || "moderate"}
- Duration: ${durationMs}ms (transition={{ duration: ${durationMs / 1000} }})
- Entrance: ${anim.entranceType || "slide"}`;
        
        // Provide animation presets based on settings
        if (anim.entranceType === "fade") {
          systemContent += `\n- Use: initial={{ opacity: 0 }} animate={{ opacity: 1 }}`;
        } else if (anim.entranceType === "slide") {
          systemContent += `\n- Use: initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}`;
        } else if (anim.entranceType === "scale") {
          systemContent += `\n- Use: initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}`;
        }
      } else {
        systemContent += `\n\n### Animation Settings
- Animations DISABLED - do not use Framer Motion animations`;
      }
    }

    if (vd.componentPreferences) {
      const cp = vd.componentPreferences;
      systemContent += `\n\n### Component Style Preferences`;
      if (cp.headerStyle) {
        systemContent += `\n- Headers: ${cp.headerStyle} variant`;
      }
      if (cp.questionStyle) {
        systemContent += `\n- Questions: ${cp.questionStyle} variant`;
      }
      if (cp.ctaStyle) {
        systemContent += `\n- CTAs/Buttons: ${cp.ctaStyle} style`;
      }
      if (cp.textStyle) {
        systemContent += `\n- Text Blocks: ${cp.textStyle} variant`;
      }
    }

    systemContent += `\n\n**IMPORTANT**: Apply these design guidelines in your JSX code! Use the specified colors, fonts, animations, and component styles consistently.`;
  }

  // Add current preview state if available
  if (state.previewState) {
    const previewCount = state.previewState.generatedPreviews?.length || 0;
    systemContent += `\n\n## Current Preview State
- Generated Previews: ${previewCount}
- Preview Mode: ${state.previewState.previewMode}
- Current Node: ${state.previewState.currentNodeId || "None selected"}`;
  }

  // Add builder progress context for continuity across orchestrator round-trips
  if (state.builderProgress) {
    const bp = state.builderProgress;
    systemContent += `\n\n## Your Previous Progress (DO NOT REPEAT THESE STEPS)

**Current Workflow Phase**: ${bp.workflow}
**Current Scope**: ${bp.currentScope || "None"}
**Generated Nodes**: ${bp.generatedNodeIds.length} nodes have previews
**Pending Nodes**: ${bp.pendingNodeIds.length} nodes awaiting generation

${bp.generatedNodeIds.length > 0 ? `**Already Generated (don't regenerate):**
${bp.generatedNodeIds.slice(-10).map(id => `- ${id}`).join("\n")}` : ""}

${bp.pendingNodeIds.length > 0 ? `**Still Need Generation:**
${bp.pendingNodeIds.slice(0, 10).map(id => `- ${id}`).join("\n")}${bp.pendingNodeIds.length > 10 ? `\n... and ${bp.pendingNodeIds.length - 10} more` : ""}` : ""}

**Recent Actions Taken**:
${bp.toolCallSummary.slice(-5).map(s => `- ${s}`).join("\n") || "No actions recorded yet"}

**IMPORTANT**: Continue from where you left off. Don't re-generate previews for nodes that already have them.`;
  }

  // Add LXD node data cache if available (for batch operations)
  if (state.builderProgress?.nodeDataCache) {
    const cache = state.builderProgress.nodeDataCache;
    const cacheEntries = Object.entries(cache).slice(0, 10);
    if (cacheEntries.length > 0) {
      systemContent += `\n\n## Cached Node Data (for batch generation)
${cacheEntries.map(([nodeId, data]) => 
  `- ${nodeId}: ${data.title} (${data.contentBlockType || 'unknown'}, intent: ${data.pedagogicalIntent || 'none'})`
).join("\n")}`;
    }
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = builderAgentTools.length > 0
    ? builderAgentModel.bindTools(builderAgentTools)
    : builderAgentModel;

  // USE BUILDER-SPECIFIC MESSAGE CHANNEL for context preservation
  // This preserves the builder's conversation history across orchestrator round-trips
  let builderConversation: BaseMessage[] = [];
  
  if (state.builderMessages && state.builderMessages.length > 0) {
    // Use builder's own message channel (already filtered and maintained)
    // CRITICAL: Must deduplicate tool_use IDs first - accumulated messages can have duplicates
    // Then filter orphaned results and repair dangling tool calls
    let deduplicated = deduplicateToolUseIds(state.builderMessages, "[builder-agent]");
    let filtered = filterOrphanedToolResults(deduplicated, "[builder-agent]");
    builderConversation = repairDanglingToolCalls(filtered, "[builder-agent]");
    console.log(`  Using ${builderConversation.length} messages from builderMessages channel (from ${state.builderMessages.length})`);
  } else {
    // First invocation or fresh start - use recent messages from main channel
    const strippedMessages = stripThinkingBlocks(state.messages || []);
    const slicedMessages = strippedMessages.slice(-15);
    // Also deduplicate in case main channel has duplicates
    let deduplicated = deduplicateToolUseIds(slicedMessages, "[builder-agent]");
    builderConversation = filterOrphanedToolResults(deduplicated, "[builder-agent]");
    console.log(`  First invocation - using ${builderConversation.length} messages from main channel`);
  }

  console.log("  Invoking builder agent model...");

  // Configure CopilotKit for proper tool emission (emits tool calls to frontend)
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });

  let response = await modelWithTools.invoke(
    [systemMessage, ...builderConversation],
    customConfig
  );

  console.log("  Builder agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:

1. Get node details with getNodeDetails(nodeId) if you haven't already
2. Write a COMPLETE React component definition like:
   const TitleBlock = () => {
     return (
       <motion.div className="...">
         ...content...
       </motion.div>
     );
   };
3. Call generateCustomComponent (single) or batchGenerateComponents (multiple) with:
   - nodeId: the node's ID
   - baseType: component type (e.g., "TitleBlock")
   - variant: layout variant (e.g., "hero") - use LXD metadata to choose!
   - jsxCode: your COMPLETE component code

CRITICAL: Your jsxCode MUST start with "const ComponentName = () =>" - never just "return" or raw JSX.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...builderConversation, nudgeMessage],
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

  // Extract component generation calls (both single and batch) to store in state
  // This stores the JSX separately so it can be stripped from message history
  const generatedComponents: GeneratedComponent[] = [];
  const newGeneratedNodeIds: string[] = [];
  const toolCallSummaries: string[] = [];
  
  if (aiResponse.tool_calls?.length) {
    for (const tc of aiResponse.tool_calls) {
      // Handle single component generation
      if (tc.name === "generateCustomComponent" && tc.args?.nodeId && tc.args?.jsxCode) {
        const existingVersion = (state.generatedComponents || []).find(
          c => c.nodeId === tc.args.nodeId
        )?.version || 0;
        
        generatedComponents.push({
          nodeId: tc.args.nodeId,
          baseType: tc.args.baseType || "Unknown",
          variant: tc.args.variant || "standard",
          jsxCode: tc.args.jsxCode,
          animationConfig: tc.args.animationConfig,
          generatedAt: new Date().toISOString(),
          version: existingVersion + 1,
        });
        
        newGeneratedNodeIds.push(tc.args.nodeId);
        toolCallSummaries.push(`Generated ${tc.args.baseType}/${tc.args.variant} for ${tc.args.nodeId.substring(0, 8)}...`);
        console.log(`  [STORE] Storing component for node ${tc.args.nodeId} (${tc.args.baseType}/${tc.args.variant})`);
      }
      
      // Handle batch component generation
      if (tc.name === "batchGenerateComponents" && tc.args?.components) {
        const components = tc.args.components as Array<{
          nodeId: string;
          baseType: string;
          variant: string;
          jsxCode: string;
          animationConfig?: any;
        }>;
        
        for (const comp of components) {
          if (comp.nodeId && comp.jsxCode) {
            const existingVersion = (state.generatedComponents || []).find(
              c => c.nodeId === comp.nodeId
            )?.version || 0;
            
            generatedComponents.push({
              nodeId: comp.nodeId,
              baseType: comp.baseType || "Unknown",
              variant: comp.variant || "standard",
              jsxCode: comp.jsxCode,
              animationConfig: comp.animationConfig,
              generatedAt: new Date().toISOString(),
              version: existingVersion + 1,
            });
            
            newGeneratedNodeIds.push(comp.nodeId);
          }
        }
        
        toolCallSummaries.push(`Batch generated ${components.length} components`);
        console.log(`  [BATCH] Storing ${components.length} components from batch call`);
      }
      
      // Track other tool calls for context
      if (tc.name === "getNodeChildren") {
        toolCallSummaries.push(`Fetched children${tc.args?.nodeId ? ` of ${String(tc.args.nodeId).substring(0, 8)}...` : ""}`);
      }
      if (tc.name === "getNodeDetails") {
        toolCallSummaries.push(`Got details for ${tc.args?.nodeId ? String(tc.args.nodeId).substring(0, 8) + "..." : "node"}`);
      }
    }
  }

  // Determine workflow phase based on activity
  let workflowPhase: BuilderProgress["workflow"] = state.builderProgress?.workflow || "idle";
  if (newGeneratedNodeIds.length > 0) {
    workflowPhase = "generating";
  }
  
  // Check if we're done (user indicated satisfaction or said [DONE])
  const responseText = typeof aiResponse.content === "string"
    ? aiResponse.content
    : Array.isArray(aiResponse.content)
    ? aiResponse.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";
  
  if (responseText.toLowerCase().includes("[done]")) {
    workflowPhase = "complete";
  }

  // Build updated progress state
  const builderProgressUpdate: BuilderProgress = {
    workflow: workflowPhase,
    currentScope: state.builderProgress?.currentScope || null,
    generatedNodeIds: newGeneratedNodeIds,
    pendingNodeIds: (state.builderProgress?.pendingNodeIds || []).filter(
      id => !newGeneratedNodeIds.includes(id)
    ),
    toolCallSummary: toolCallSummaries,
    nodeDataCache: state.builderProgress?.nodeDataCache,
    lastUpdated: new Date().toISOString(),
  };

  return {
    messages: [response],
    currentAgent: "builder_agent",
    agentHistory: ["builder_agent"],
    // Clear routing decision when this agent starts
    routingDecision: null,
    // Store generated components separately from messages
    // This allows jsxCode to be stripped from message history while preserving the data
    ...(generatedComponents.length > 0 ? { generatedComponents } : {}),
    // CRITICAL: Append to builder's own message channel for continuity
    builderMessages: [response],
    // Update builder progress state for semantic context
    builderProgress: builderProgressUpdate,
  };
}

export default builderAgentNode;

