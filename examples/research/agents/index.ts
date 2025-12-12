/**
 * Agent Exports
 *
 * Barrel export for all sub-agents in the orchestrator system.
 * 
 * Architecture (CopilotKit StateGraph pattern):
 * - Each agent node filters tools from state.copilotkit?.actions
 * - Binds filtered tools to its model
 * - Tool calls route to END for CopilotKit to execute on frontend
 * - Each agent can stream to CopilotKit independently
 */

// Original creative workflow agents
export { strategistNode, parseProjectBrief } from "./strategist";
export { researcherNode, researcherTools, parseResearchFindings } from "./researcher";
export { architectNode, parseCourseStructure, structureToTree } from "./architect";
export { writerNode, extractContentOutput, getNextNodeToWrite, generateWritingBrief } from "./writer";
export {
  visualDesignerNode,
  parseVisualDesign,
  getThemePreset,
  suggestThemes,
  DESIGN_THEMES,
} from "./visual-designer";
export { builderAgentNode } from "./builder-agent";

// Tool-specialized sub-agents (for reduced payload architecture)
// Each filters CopilotKit tools to its specific subset
export { dataAgentNode } from "./data-agent";
export { projectAgentNode } from "./project-agent";
export { nodeAgentNode } from "./node-agent";
export { documentAgentNode } from "./document-agent";
export { mediaAgentNode } from "./media-agent";
export { frameworkAgentNode } from "./framework-agent";

// Image Generator (compiled StateGraph as tool)
// Used by Writer for batch image generation with context isolation
export { imageGeneratorTool, imageGeneratorGraph } from "./image-generator";

