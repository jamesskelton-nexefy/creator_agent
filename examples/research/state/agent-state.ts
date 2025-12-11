/**
 * Shared State Definitions for Multi-Agent Orchestrator
 *
 * Defines the state annotation that flows between all agents in the system.
 * Each agent reads from and writes to specific keys in this shared state.
 */

import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { CopilotKitStateAnnotation } from "@copilotkit/sdk-js/langgraph";
import { BaseMessage } from "@langchain/core/messages";

// ============================================================================
// BOUNDED MESSAGES CONFIGURATION
// ============================================================================

/**
 * Maximum number of messages to keep in state.
 * This prevents "RangeError: Invalid string length" during JSON serialization
 * by FileSystemPersistence (langgraph-cli dev mode).
 * 
 * CRITICAL: Keep this LOW to prevent state explosion during serialization.
 * Tool results can be very large (templates, hierarchy info), so even a 
 * moderate number of messages can exceed JSON.stringify limits.
 * 
 * Note: Context management (processContext) handles LLM context separately.
 * This limit is specifically for state persistence.
 */
const MAX_MESSAGES_IN_STATE = 40;

/**
 * Maximum number of written content entries to keep.
 * Prevents unbounded accumulation in long sessions.
 */
const MAX_WRITTEN_CONTENT = 50;

/**
 * Maximum number of created nodes to track.
 * Prevents unbounded accumulation in long sessions.
 */
const MAX_CREATED_NODES = 50;

/**
 * Maximum number of generated previews to keep.
 * Prevents unbounded accumulation in long sessions.
 */
const MAX_GENERATED_PREVIEWS = 20;

// ============================================================================
// TYPE DEFINITIONS - Structured outputs from each agent
// ============================================================================

/**
 * Project brief from the Strategist agent.
 * Captures purpose, objectives, scope, and constraints.
 */
export interface ProjectBrief {
  /** The primary purpose/goal of the training */
  purpose: string;
  /** Specific learning objectives */
  objectives: string[];
  /** What is in scope for this project */
  inScope: string[];
  /** What is explicitly out of scope */
  outOfScope: string[];
  /** Constraints and considerations */
  constraints: string[];
  /** Target audience/learner persona */
  targetAudience: string;
  /** Industry context */
  industry: string;
  /** Any regulatory requirements */
  regulations?: string[];
  /** Relevant frameworks, training packages, or competency standards */
  frameworks?: {
    /** Framework ID if linked to a project */
    id?: string;
    /** Name of framework/training package (e.g., "TLI Transport and Logistics") */
    name: string;
    /** Type: "training_package", "asqa_unit", "custom", "uploaded" */
    type: string;
    /** Specific units or competencies of interest */
    units?: string[];
    /** Notes about relevance */
    notes?: string;
  }[];
  /** Additional notes from user clarifications */
  notes?: string;
}

/**
 * Research findings from the Researcher agent.
 * Contains deep knowledge gathered from web search and documents.
 */
export interface ResearchBrief {
  /** Industry overview and context */
  industryContext: string;
  /** Key topics and concepts to cover */
  keyTopics: {
    topic: string;
    summary: string;
    importance: "critical" | "important" | "supplementary";
  }[];
  /** Relevant regulations and compliance requirements */
  regulations: {
    name: string;
    summary: string;
    relevance: string;
  }[];
  /** Learner persona insights */
  personaInsights: string;
  /** Best practices for the industry */
  bestPractices: string[];
  /** Sources and citations */
  citations: {
    title: string;
    url?: string;
    summary: string;
  }[];
  /** Raw research notes for reference */
  rawNotes?: string;
}

/**
 * Planned structure from the Architect agent (PRE-creation phase).
 * This is the plan BEFORE nodes are created, allowing resumption if interrupted.
 * Separate from CourseStructure which represents the FINAL created structure.
 */
export interface PlannedStructure {
  /** Overall structure summary */
  summary: string;
  /** Planned nodes in hierarchical order */
  nodes: PlannedNode[];
  /** Estimated total content pieces */
  totalNodes: number;
  /** Hierarchy depth (number of levels used) */
  maxDepth: number;
  /** Rationale for the structure */
  rationale: string;
  /** Planning timestamp */
  plannedAt: string;
  /** Status of plan execution */
  executionStatus: "planned" | "in_progress" | "completed" | "failed";
  /** Nodes that have been created (maps tempId to actual nodeId) */
  executedNodes: Record<string, string>;
}

/**
 * Course structure from the Architect agent.
 * Detailed hierarchy of planned nodes.
 */
export interface CourseStructure {
  /** Overall structure summary */
  summary: string;
  /** Planned nodes in hierarchical order */
  nodes: PlannedNode[];
  /** Estimated total content pieces */
  totalNodes: number;
  /** Hierarchy depth (number of levels used) */
  maxDepth: number;
  /** Rationale for the structure */
  rationale: string;
}

/**
 * A planned node in the course structure.
 */
export interface PlannedNode {
  /** Temporary ID for reference */
  tempId: string;
  /** Node title */
  title: string;
  /** Node type (module, lesson, topic, etc.) */
  nodeType: string;
  /** Hierarchy level (2-6) */
  level: number;
  /** Parent temp ID (null for top-level) */
  parentTempId: string | null;
  /** Template ID to use */
  templateId?: string;
  /** Brief description of content */
  description: string;
  /** Learning objectives for this node */
  objectives?: string[];
  /** Order among siblings */
  orderIndex: number;
}

/**
 * Image style configuration for a specific image type.
 */
export interface ImageStyleConfig {
  /** Image type: photo, illustration, 3d, icon */
  type: "photo" | "illustration" | "3d" | "icon";
  /** Style variant within the type */
  style: string;
  /** Additional style settings */
  lighting?: string;
  mood?: string;
  complexity?: "simple" | "medium" | "detailed";
}

/**
 * Extended color system beyond basic palette.
 */
export interface ColorSystemConfig {
  /** Primary gradient (Tailwind format) */
  gradientPrimary?: string;
  /** Secondary gradient */
  gradientSecondary?: string;
  /** Card/surface background color */
  surfaceCard?: string;
  /** Elevated surface color */
  surfaceElevated?: string;
  /** Shadow intensity level */
  shadowIntensity?: "none" | "subtle" | "medium" | "strong";
}

/**
 * Layout and spacing preferences.
 */
export interface LayoutConfig {
  /** Container padding level */
  containerPadding: "compact" | "normal" | "spacious";
  /** Border radius style */
  borderRadius: "none" | "subtle" | "rounded" | "pill";
  /** Card component style */
  cardStyle: "flat" | "elevated" | "outlined" | "glass";
}

/**
 * Animation preferences for components.
 */
export interface AnimationConfig {
  /** Whether animations are enabled */
  enabled: boolean;
  /** Animation intensity */
  style?: "subtle" | "moderate" | "dynamic";
  /** Animation speed */
  duration?: "fast" | "normal" | "slow";
  /** Entrance animation type */
  entranceType?: "fade" | "slide" | "scale";
  /** Duration in milliseconds */
  durationMs?: number;
}

/**
 * Component-level style preferences.
 */
export interface ComponentPreferences {
  /** Header/title block style */
  headerStyle?: "minimal" | "gradient" | "hero" | "split";
  /** Question block style */
  questionStyle?: "standard" | "card" | "gamified";
  /** CTA/action button style */
  ctaStyle?: "solid" | "outline" | "gradient";
  /** Text block style */
  textStyle?: "standard" | "callout" | "quote" | "two-column";
}

/**
 * Visual design specification from the Visual Designer agent.
 */
export interface VisualDesign {
  /** Selected visual theme/style */
  theme: string;
  /** Color palette */
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    text: string;
  };
  /** Typography choices */
  typography: {
    headingFont: string;
    bodyFont: string;
    style: "formal" | "casual" | "technical" | "friendly";
  };
  /** Writing tone guidelines */
  writingTone: {
    tone: "professional" | "conversational" | "academic" | "engaging";
    voice: "first-person" | "second-person" | "third-person";
    complexity: "simple" | "intermediate" | "advanced";
  };
  /** Brand elements if provided */
  branding?: {
    logoUrl?: string;
    companyName?: string;
    brandGuidelines?: string;
  };
  /** Additional style notes */
  notes?: string;
  
  // ========== EXTENDED DESIGN SYSTEM ==========
  
  /**
   * Image styles - array to support multiple types (e.g., photos AND illustrations).
   * Each entry specifies a type and the selected style variant within that type.
   */
  imageStyles?: ImageStyleConfig[];
  
  /**
   * Extended color system with gradients and surfaces.
   */
  colorSystem?: ColorSystemConfig;
  
  /**
   * Layout and spacing preferences.
   */
  layout?: LayoutConfig;
  
  /**
   * Animation preferences for components.
   */
  animation?: AnimationConfig;
  
  /**
   * Component-level style preferences.
   */
  componentPreferences?: ComponentPreferences;
}

/**
 * Generated preview component from Builder Agent.
 * Represents a rendered e-learning component for a content node.
 */
export interface GeneratedPreview {
  /** The content node ID this preview is for */
  nodeId: string;
  /** Component type (TitleBlock, QuestionBlock, etc.) */
  componentType: string;
  /** Props to pass to the component */
  props: Record<string, any>;
  /** Optional raw HTML output */
  html?: string;
  /** Optional CSS styles */
  css?: string;
  /** When this preview was generated */
  generatedAt: string;
  /** Version number (incremented on updates) */
  version: number;
  /** User instructions that led to this version */
  userInstructions?: string;
}

/**
 * Preview generation state from Builder Agent.
 * Tracks all generated previews and current preview mode.
 */
export interface PreviewState {
  /** All generated previews indexed by nodeId */
  generatedPreviews: GeneratedPreview[];
  /** Currently focused node ID */
  currentNodeId: string | null;
  /** Preview mode: single node or flow of nodes */
  previewMode: 'single' | 'flow';
  /** Last update timestamp */
  lastUpdated: string;
}

/**
 * Content output tracking from the Writer agent.
 */
export interface ContentOutput {
  /** The created node ID */
  nodeId: string;
  /** The temp ID from the plan */
  tempId: string;
  /** Node title */
  title: string;
  /** Fields that were populated */
  fieldsWritten: string[];
  /** Creation timestamp */
  createdAt: string;
  /** Any issues or notes */
  notes?: string;
}

/**
 * Node snapshot summary - lightweight cache of getNodeTreeSnapshot results
 * Stored in progress state to avoid repeated full snapshot calls
 */
export interface NodeSnapshotSummary {
  /** Total nodes in project */
  totalNodes: number;
  /** Nodes by level (level number -> count) */
  nodesByLevel: Record<number, number>;
  /** Content Blocks that need content (IDs) */
  contentBlocksNeedingContent: string[];
  /** Content Blocks that have content (count only to save space) */
  contentBlocksWithContentCount: number;
  /** When snapshot was taken */
  snapshotAt: string;
}

/**
 * Writer progress state - maintains context across orchestrator round-trips.
 * This is critical for long-running writer sessions where context trimming
 * would otherwise cause the writer to lose track of its work.
 */
export interface WriterProgress {
  /** Nodes the writer has already explored/navigated to */
  exploredNodes: string[];
  /** Current parent node ID the writer is working under */
  currentParentId: string | null;
  /** Current workflow phase */
  workflow: "exploring" | "creating" | "reviewing";
  /** Summary of recent tool calls for context */
  toolCallSummary: string[];
  /** Hierarchy info cached from exploration (to avoid re-fetching) */
  hierarchyCache?: {
    levelNames: string[];
    maxDepth: number;
    contentLevel: number;
  };
  /** Node snapshot summary (from getNodeTreeSnapshot) */
  nodeSnapshot?: NodeSnapshotSummary;
  /** Last update timestamp */
  lastUpdated: string;
}

/**
 * Architect progress state - maintains context across orchestrator round-trips.
 * This is critical for long-running architect sessions where context trimming
 * would otherwise cause the architect to lose track of its node creation work.
 */
export interface ArchitectProgress {
  /** Current workflow phase */
  workflow: "planning" | "awaiting_approval" | "building" | "complete";
  /** Nodes the architect has already explored/navigated to */
  exploredNodes: string[];
  /** Summary of recent tool calls for context */
  toolCallSummary: string[];
  /** Hierarchy info cached from exploration (to avoid re-fetching) */
  hierarchyCache?: {
    levelNames: string[];
    maxDepth: number;
    templateIds: Record<string, string>; // nodeType -> templateId mapping
  };
  /** Node snapshot summary (from getNodeTreeSnapshot) */
  nodeSnapshot?: NodeSnapshotSummary;
  /** TempId to actual nodeId mapping for created nodes */
  tempIdToNodeId: Record<string, string>;
  /** Count of nodes created in this session */
  nodesCreatedCount: number;
  /** Last update timestamp */
  lastUpdated: string;
}

/**
 * Tracks a node that has been created in this session.
 * Used to prevent duplicate creation.
 */
export interface CreatedNode {
  parentNodeId: string | null;
  title: string;
  nodeId: string;
  templateName: string;
}

/**
 * Identifies which agent should be invoked next.
 * 
 * Agents are divided into:
 * - Creative workflow: strategist, researcher, architect, writer, visual_designer
 * - Tool-specialized: project_agent, node_agent, data_agent, document_agent, media_agent, framework_agent
 */
export type AgentType = 
  | "orchestrator"
  // Creative workflow agents
  | "strategist" 
  | "researcher" 
  | "architect" 
  | "writer" 
  | "visual_designer"
  | "builder_agent"
  // Tool-specialized sub-agents
  | "project_agent"
  | "node_agent"
  | "data_agent"
  | "document_agent"
  | "media_agent"
  | "framework_agent";

/**
 * Routing decision made by the orchestrator.
 */
export interface RoutingDecision {
  /** Which agent to invoke */
  nextAgent: AgentType;
  /** Reason for the routing decision */
  reason: string;
  /** Specific task/instructions for the agent */
  task: string;
}

// ============================================================================
// AGENT WORK STATE - Phase-based workflow control
// ============================================================================

/**
 * Tracks an agent's current work state including phase and allowed tools.
 * This enables phase-gated workflows where agents must complete one phase
 * before moving to the next (e.g., gather requirements → search → create brief).
 */
export interface AgentWorkState {
  /** Which agent is doing work */
  agent: AgentType;
  /** Current phase of the agent's workflow */
  phase: string;
  /** Tool that triggered the wait (if HITL) */
  pendingTool?: string;
  /** Tools the agent is allowed to use in this phase */
  allowedTools?: string[];
  /** Additional metadata for the phase */
  metadata?: {
    questionsCount?: number;
    searchesCompleted?: number;
    optionsPresented?: number;
    [key: string]: unknown;
  };
}

/**
 * Tracks the active task/goal being worked on.
 * This persists across agent transitions to maintain context about:
 * - What the user originally asked for
 * - What we're trying to accomplish
 * - Progress made so far
 * 
 * Essential for long-running threads where agents need to remember
 * the current task even after context trimming occurs.
 */
export interface ActiveTask {
  /** The user's original request that initiated this task */
  originalRequest: string;
  /** High-level description of what we're trying to accomplish */
  currentGoal: string;
  /** Which agent is currently responsible for this task */
  assignedAgent: AgentType;
  /** List of completed steps/milestones */
  progress: string[];
  /** When the task was started */
  startedAt: string;
  /** Optional: specific instructions passed to the assigned agent */
  agentInstructions?: string;
  /** Optional: expected next steps after current agent completes */
  nextSteps?: string[];
}

/**
 * Phase definitions for the Strategist agent.
 * Controls workflow: gather requirements → search references → create brief
 */
export const STRATEGIST_PHASES = {
  gathering_requirements: {
    description: "Asking clarifying questions to understand the project",
    allowedTools: ["askClarifyingQuestions", "offerOptions", "listDocuments"],
    nextPhase: "searching_references",
  },
  searching_references: {
    description: "Searching for relevant frameworks, units, and standards",
    allowedTools: ["searchASQAUnits", "listFrameworks", "getFrameworkDetails", "importASQAUnit"],
    nextPhase: "creating_brief",
  },
  creating_brief: {
    description: "Creating the final project brief - no tool calls allowed",
    allowedTools: [],
    nextPhase: "complete",
  },
  complete: {
    description: "Work complete",
    allowedTools: [],
    nextPhase: null,
  },
} as const;

/**
 * Phase definitions for the Visual Designer agent.
 * Controls workflow:
 *   gather preferences → select image types → select styles → 
 *   select colors/typography → select animations/components → finalize
 */
export const VISUAL_DESIGNER_PHASES = {
  gathering_preferences: {
    description: "Understanding basic design preferences and requirements",
    allowedTools: ["offerOptions", "askClarifyingQuestions", "searchMicroverse"],
    nextPhase: "selecting_image_types",
  },
  selecting_image_types: {
    description: "User selects which image types to use (photos, illustrations, 3D, icons)",
    allowedTools: ["selectImageTypes"],
    nextPhase: "selecting_image_styles",
  },
  selecting_image_styles: {
    description: "User selects specific styles for each chosen image type",
    allowedTools: ["selectAllImageStyles"],
    nextPhase: "selecting_colors",
  },
  selecting_colors: {
    description: "User selects color palette",
    allowedTools: ["selectColorPalette"],
    nextPhase: "selecting_typography",
  },
  selecting_typography: {
    description: "User selects typography/font pairing",
    allowedTools: ["selectTypography"],
    nextPhase: "selecting_animations",
  },
  selecting_animations: {
    description: "User selects animation style",
    allowedTools: ["selectAnimationStyle"],
    nextPhase: "selecting_components",
  },
  selecting_components: {
    description: "User selects component style preferences",
    allowedTools: ["selectComponentStyles"],
    nextPhase: "finalizing",
  },
  finalizing: {
    description: "Finalizing the design specification - compiling all selections",
    allowedTools: [],
    nextPhase: "complete",
  },
  complete: {
    description: "Work complete",
    allowedTools: [],
    nextPhase: null,
  },
} as const;

/**
 * Phase definitions for the Researcher agent.
 * Controls workflow: research → synthesize findings
 */
export const RESEARCHER_PHASES = {
  researching: {
    description: "Gathering information from web and documents",
    allowedTools: ["web_search", "listDocuments", "searchDocuments", "searchDocumentsByText", "getDocumentLines", "getDocumentByName", "searchMicroverse"],
    nextPhase: "synthesizing",
  },
  synthesizing: {
    description: "Synthesizing research findings - no tool calls allowed",
    allowedTools: [],
    nextPhase: "complete",
  },
  complete: {
    description: "Work complete",
    allowedTools: [],
    nextPhase: null,
  },
} as const;

/**
 * Phase definitions for the Architect agent.
 * Controls workflow: analyze structure → design course
 */
export const ARCHITECT_PHASES = {
  analyzing: {
    description: "Analyzing project structure and requirements",
    allowedTools: ["getProjectHierarchyInfo", "getNodesByLevel", "getNodeChildren", "getNodeDetails", "getAvailableTemplates", "listAllNodeTemplates"],
    nextPhase: "designing",
  },
  designing: {
    description: "Designing and creating the course structure",
    allowedTools: ["requestEditMode", "releaseEditMode", "createNode", "getAvailableTemplates", "getNodeTemplateFields", "offerOptions", "requestPlanApproval"],
    nextPhase: "complete",
  },
  complete: {
    description: "Work complete",
    allowedTools: [],
    nextPhase: null,
  },
} as const;

/**
 * Phase definitions for the Writer agent.
 * Controls workflow: plan content → write content
 */
export const WRITER_PHASES = {
  planning: {
    description: "Planning content structure and gathering context",
    allowedTools: ["getNodesByLevel", "getNodeDetails", "getNodeChildren", "getAvailableTemplates", "searchDocuments", "searchMicroverse"],
    nextPhase: "writing",
  },
  writing: {
    description: "Writing and creating content nodes",
    allowedTools: ["requestEditMode", "releaseEditMode", "createNode", "updateNodeFields", "getNodeTemplateFields", "attachMicroverseToNode"],
    nextPhase: "complete",
  },
  complete: {
    description: "Work complete",
    allowedTools: [],
    nextPhase: null,
  },
} as const;

/** Type for phase keys */
export type StrategistPhase = keyof typeof STRATEGIST_PHASES;
export type VisualDesignerPhase = keyof typeof VISUAL_DESIGNER_PHASES;
export type ResearcherPhase = keyof typeof RESEARCHER_PHASES;
export type ArchitectPhase = keyof typeof ARCHITECT_PHASES;
export type WriterPhase = keyof typeof WRITER_PHASES;

// ============================================================================
// STATE ANNOTATION - Shared state for all agents
// ============================================================================

/**
 * Shared state annotation that all agents in the system use.
 * Extends CopilotKitStateAnnotation to inherit frontend tool access.
 */
export const OrchestratorStateAnnotation = Annotation.Root({
  // Inherit CopilotKit state (messages, actions, context)
  ...CopilotKitStateAnnotation.spec,

  // ---- OVERRIDE: Bounded Messages Reducer ----
  // Override the default messages reducer with one that caps at MAX_MESSAGES_IN_STATE
  // This prevents state explosion that causes "RangeError: Invalid string length"
  // during JSON serialization by FileSystemPersistence (langgraph-cli dev mode)
  messages: Annotation<BaseMessage[]>({
    reducer: (existing, update) => {
      // Use standard messagesStateReducer for accumulation
      const accumulated = messagesStateReducer(existing, update);
      // Cap at MAX_MESSAGES_IN_STATE to prevent unbounded growth
      if (accumulated.length > MAX_MESSAGES_IN_STATE) {
        // Keep most recent messages, preserving conversation flow
        return accumulated.slice(-MAX_MESSAGES_IN_STATE);
      }
      return accumulated;
    },
    default: () => [],
  }),

  // ---- Agent Context Outputs ----

  /** Project brief from the Strategist */
  projectBrief: Annotation<ProjectBrief | null>({
    reducer: (existing, update) => update ?? existing,
    default: () => null,
  }),

  /** Research findings from the Researcher */
  researchFindings: Annotation<ResearchBrief | null>({
    reducer: (existing, update) => update ?? existing,
    default: () => null,
  }),

  /** Planned structure from the Architect (PRE-creation phase) */
  plannedStructure: Annotation<PlannedStructure | null>({
    reducer: (existing, update) => {
      if (!update) return existing;
      if (!existing) return update;
      // Merge executedNodes from both
      return {
        ...update,
        executedNodes: { ...existing.executedNodes, ...update.executedNodes },
      };
    },
    default: () => null,
  }),

  /** Course structure from the Architect (FINAL created structure) */
  courseStructure: Annotation<CourseStructure | null>({
    reducer: (existing, update) => update ?? existing,
    default: () => null,
  }),

  /** Visual design from the Visual Designer */
  visualDesign: Annotation<VisualDesign | null>({
    reducer: (existing, update) => update ?? existing,
    default: () => null,
  }),

  /** Preview state from the Builder Agent (previews capped at MAX_GENERATED_PREVIEWS) */
  previewState: Annotation<PreviewState | null>({
    reducer: (existing, update) => {
      if (!update) return existing;
      if (!existing) return update;
      // Merge generated previews, keeping newer versions
      let mergedPreviews = [...existing.generatedPreviews];
      for (const preview of update.generatedPreviews || []) {
        const existingIdx = mergedPreviews.findIndex(p => p.nodeId === preview.nodeId);
        if (existingIdx >= 0) {
          // Keep newer version
          if (preview.version > mergedPreviews[existingIdx].version) {
            mergedPreviews[existingIdx] = preview;
          }
        } else {
          mergedPreviews.push(preview);
        }
      }
      // Cap at MAX_GENERATED_PREVIEWS to prevent unbounded growth
      if (mergedPreviews.length > MAX_GENERATED_PREVIEWS) {
        mergedPreviews = mergedPreviews.slice(-MAX_GENERATED_PREVIEWS);
      }
      return {
        ...update,
        generatedPreviews: mergedPreviews,
      };
    },
    default: () => null,
  }),

  /** Content outputs from the Writer (accumulates, capped at MAX_WRITTEN_CONTENT) */
  writtenContent: Annotation<ContentOutput[]>({
    reducer: (existing, update) => {
      // Merge new content with existing, avoiding duplicates by nodeId
      const merged = [...(existing || [])];
      for (const content of update || []) {
        if (!merged.some((c) => c.nodeId === content.nodeId)) {
          merged.push(content);
        }
      }
      // Cap at MAX_WRITTEN_CONTENT to prevent unbounded growth
      if (merged.length > MAX_WRITTEN_CONTENT) {
        return merged.slice(-MAX_WRITTEN_CONTENT);
      }
      return merged;
    },
    default: () => [],
  }),

  /**
   * Writer-specific message channel.
   * Persists the writer's conversation independently from the main messages channel.
   * This prevents context loss when the writer hands back to orchestrator and returns.
   * Capped at 50 messages to prevent unbounded growth.
   */
  writerMessages: Annotation<BaseMessage[]>({
    reducer: (existing, update) => {
      // Accumulate messages
      const accumulated = [...(existing || []), ...(update || [])];
      // Cap at 50 messages to prevent state explosion
      if (accumulated.length > 50) {
        return accumulated.slice(-50);
      }
      return accumulated;
    },
    default: () => [],
  }),

  /**
   * Writer progress state - semantic context for writer continuity.
   * Tracks what the writer has explored and created, surviving orchestrator round-trips.
   */
  writerProgress: Annotation<WriterProgress | null>({
    reducer: (existing, update) => {
      if (update === undefined) return existing;
      if (update === null) return null;
      if (!existing) return update;
      // Merge: accumulate explored nodes and tool summaries
      const mergedExplored = [...new Set([...(existing.exploredNodes || []), ...(update.exploredNodes || [])])];
      const mergedToolSummary = [...(existing.toolCallSummary || []), ...(update.toolCallSummary || [])].slice(-20);
      return {
        ...existing,
        ...update,
        exploredNodes: mergedExplored.slice(-50), // Cap explored nodes
        toolCallSummary: mergedToolSummary,
      };
    },
    default: () => null,
  }),

  /**
   * Architect-specific message channel.
   * Persists the architect's conversation independently from the main messages channel.
   * This prevents context loss when the architect hands back to orchestrator and returns.
   * Capped at 50 messages to prevent unbounded growth.
   */
  architectMessages: Annotation<BaseMessage[]>({
    reducer: (existing, update) => {
      // Accumulate messages
      const accumulated = [...(existing || []), ...(update || [])];
      // Cap at 50 messages to prevent state explosion
      if (accumulated.length > 50) {
        return accumulated.slice(-50);
      }
      return accumulated;
    },
    default: () => [],
  }),

  /**
   * Architect progress state - semantic context for architect continuity.
   * Tracks what the architect has explored and created, surviving orchestrator round-trips.
   * Critical for maintaining tempId -> nodeId mappings during node creation.
   */
  architectProgress: Annotation<ArchitectProgress | null>({
    reducer: (existing, update) => {
      if (update === undefined) return existing;
      if (update === null) return null;
      if (!existing) return update;
      // Merge: accumulate explored nodes, tool summaries, and tempId mappings
      const mergedExplored = [...new Set([...(existing.exploredNodes || []), ...(update.exploredNodes || [])])];
      const mergedToolSummary = [...(existing.toolCallSummary || []), ...(update.toolCallSummary || [])].slice(-20);
      const mergedTempIdMap = { ...(existing.tempIdToNodeId || {}), ...(update.tempIdToNodeId || {}) };
      return {
        ...existing,
        ...update,
        exploredNodes: mergedExplored.slice(-50), // Cap explored nodes
        toolCallSummary: mergedToolSummary,
        tempIdToNodeId: mergedTempIdMap,
        nodesCreatedCount: (existing.nodesCreatedCount || 0) + (update.nodesCreatedCount || 0),
      };
    },
    default: () => null,
  }),

  // ---- Execution Tracking ----

  /** Track created nodes to prevent duplicates (capped at MAX_CREATED_NODES) */
  createdNodes: Annotation<CreatedNode[]>({
    reducer: (existing, update) => {
      const merged = [...(existing || [])];
      for (const node of update || []) {
        if (!merged.some((n) => n.nodeId === node.nodeId)) {
          merged.push(node);
        }
      }
      // Cap at MAX_CREATED_NODES to prevent unbounded growth
      if (merged.length > MAX_CREATED_NODES) {
        return merged.slice(-MAX_CREATED_NODES);
      }
      return merged;
    },
    default: () => [],
  }),

  /** Current routing decision */
  routingDecision: Annotation<RoutingDecision | null>({
    // FIX: Use update directly so null actually clears the value
    // Previously `update ?? existing` would keep old value when update was null
    reducer: (existing, update) => update === undefined ? existing : update,
    default: () => null,
  }),

  /** 
   * Tracks an agent's current work state including phase and allowed tools.
   * Enables phase-gated workflows where agents must complete one phase
   * before moving to the next (e.g., gather requirements → search → create brief).
   * 
   * When set, the supervisor will route back to this agent until work is complete.
   * Agents should update the phase as they progress through their workflow.
   */
  awaitingUserAction: Annotation<AgentWorkState | null>({
    reducer: (existing, update) => update === undefined ? existing : update,
    default: () => null,
  }),

  /**
   * Tracks the active task/goal being worked on.
   * Persists across agent transitions to maintain context about:
   * - What the user originally asked for
   * - What we're trying to accomplish
   * - Progress made so far
   * 
   * This is critical for long-running threads where context trimming
   * may remove the original user request from the message history.
   * Supervisor sets this when routing to an agent, and agents update
   * the progress array as they complete steps.
   */
  activeTask: Annotation<ActiveTask | null>({
    reducer: (existing, update) => {
      if (update === undefined) return existing;
      if (update === null) return null;
      // Merge progress arrays when updating
      if (existing && update) {
        const mergedProgress = [...(existing.progress || [])];
        for (const step of update.progress || []) {
          if (!mergedProgress.includes(step)) {
            mergedProgress.push(step);
          }
        }
        // Cap progress at 20 entries to prevent state explosion
        const cappedProgress = mergedProgress.slice(-20);
        return { ...existing, ...update, progress: cappedProgress };
      }
      return update;
    },
    default: () => null,
  }),

  /** Which agent is currently active */
  currentAgent: Annotation<AgentType>({
    reducer: (existing, update) => update ?? existing,
    default: () => "orchestrator",
  }),

  /** Tracks which agents have been invoked this session (capped at 50 entries) */
  agentHistory: Annotation<AgentType[]>({
    reducer: (existing, update) => {
      const combined = [...(existing || []), ...(update || [])];
      return combined.slice(-50); // Keep only last 50 entries to prevent state explosion
    },
    default: () => [],
  }),

  /** Error state if an agent fails */
  lastError: Annotation<string | null>({
    reducer: (existing, update) => update,
    default: () => null,
  }),

  // ---- Research Iteration Tracking ----

  /** 
   * Tracks how many research iterations have been completed.
   * Used to prevent infinite research loops.
   * Reset when research is complete or a new research task starts.
   */
  researchIterationCount: Annotation<number>({
    reducer: (existing, update) => update ?? existing,
    default: () => 0,
  }),
});

export type OrchestratorState = typeof OrchestratorStateAnnotation.State;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Creates a summary of the current agent context for logging/debugging.
 */
export function summarizeAgentContext(state: OrchestratorState): string {
  const parts: string[] = [];
  
  if (state.projectBrief) {
    parts.push(`[Brief] Purpose: ${state.projectBrief.purpose.substring(0, 50)}...`);
  }
  if (state.researchFindings) {
    parts.push(`[Research] ${state.researchFindings.keyTopics.length} topics, ${state.researchFindings.citations.length} citations`);
  }
  if (state.courseStructure) {
    parts.push(`[Structure] ${state.courseStructure.totalNodes} planned nodes`);
  }
  if (state.visualDesign) {
    parts.push(`[Design] Theme: ${state.visualDesign.theme}`);
  }
  if (state.writtenContent.length > 0) {
    parts.push(`[Written] ${state.writtenContent.length} nodes created`);
  }
  
  return parts.length > 0 ? parts.join(" | ") : "No agent context yet";
}

/**
 * Generates a task context summary for injection into agent system prompts.
 * This ensures agents always know:
 * - What the user originally requested
 * - What the current goal is
 * - What progress has been made
 * - What other agents have produced
 * 
 * This is critical for maintaining context in long-running threads where
 * message trimming may have removed the original user request.
 */
export function generateTaskContext(state: OrchestratorState): string {
  const sections: string[] = [];

  // Active task context (most important - what we're working on)
  if (state.activeTask) {
    sections.push(`## Current Task

**Original User Request:**
${state.activeTask.originalRequest}

**Current Goal:**
${state.activeTask.currentGoal}

**Assigned Agent:** ${state.activeTask.assignedAgent}
**Started:** ${state.activeTask.startedAt}
${state.activeTask.agentInstructions ? `\n**Instructions:** ${state.activeTask.agentInstructions}` : ""}
${state.activeTask.progress.length > 0 ? `\n**Progress Made:**\n${state.activeTask.progress.map(p => `- ${p}`).join("\n")}` : ""}
${state.activeTask.nextSteps?.length ? `\n**Expected Next Steps:**\n${state.activeTask.nextSteps.map(s => `- ${s}`).join("\n")}` : ""}`);
  }

  // Work already completed by other agents
  const completedWork: string[] = [];
  
  if (state.projectBrief) {
    completedWork.push(`- **Project Brief** (Strategist): Purpose is "${state.projectBrief.purpose.substring(0, 100)}...", ${state.projectBrief.objectives.length} objectives defined, targeting ${state.projectBrief.targetAudience} in ${state.projectBrief.industry}`);
  }
  
  if (state.researchFindings) {
    completedWork.push(`- **Research** (Researcher): ${state.researchFindings.keyTopics.length} key topics identified, ${state.researchFindings.citations.length} citations gathered`);
  }
  
  if (state.plannedStructure) {
    const executed = Object.keys(state.plannedStructure.executedNodes || {}).length;
    completedWork.push(`- **Course Plan** (Architect): ${state.plannedStructure.nodes.length} nodes planned, ${executed} created, status: ${state.plannedStructure.executionStatus}`);
  }
  
  if (state.courseStructure) {
    completedWork.push(`- **Course Structure** (Architect): ${state.courseStructure.totalNodes} total nodes, ${state.courseStructure.maxDepth} levels deep`);
  }
  
  if (state.visualDesign) {
    completedWork.push(`- **Visual Design** (Designer): Theme "${state.visualDesign.theme}", tone "${state.visualDesign.writingTone.tone}"`);
  }
  
  if (state.writtenContent && state.writtenContent.length > 0) {
    completedWork.push(`- **Content** (Writer): ${state.writtenContent.length} content nodes created`);
  }

  if (completedWork.length > 0) {
    sections.push(`## Completed Work\n\n${completedWork.join("\n")}`);
  }

  // Current agent work state (phase info)
  if (state.awaitingUserAction) {
    sections.push(`## Current Work State

**Agent:** ${state.awaitingUserAction.agent}
**Phase:** ${state.awaitingUserAction.phase}
${state.awaitingUserAction.pendingTool ? `**Pending Tool:** ${state.awaitingUserAction.pendingTool}` : ""}
${state.awaitingUserAction.allowedTools?.length ? `**Allowed Tools:** ${state.awaitingUserAction.allowedTools.join(", ")}` : ""}`);
  }

  return sections.length > 0 ? sections.join("\n\n") : "";
}

/**
 * Checks if a specific agent's output is available.
 */
export function hasAgentOutput(state: OrchestratorState, agent: AgentType): boolean {
  switch (agent) {
    case "strategist":
      return state.projectBrief !== null;
    case "researcher":
      return state.researchFindings !== null;
    case "architect":
      return state.courseStructure !== null;
    case "visual_designer":
      return state.visualDesign !== null;
    case "writer":
      return state.writtenContent.length > 0;
    default:
      return false;
  }
}

/**
 * Gets a condensed version of research findings for passing to other agents.
 * Reduces context size while preserving key information.
 */
export function getCondensedResearch(findings: ResearchBrief): string {
  const topicList = findings.keyTopics
    .filter((t) => t.importance !== "supplementary")
    .map((t) => `- ${t.topic}: ${t.summary}`)
    .join("\n");
  
  const regulationList = findings.regulations
    .map((r) => `- ${r.name}: ${r.summary}`)
    .join("\n");
  
  return `
## Industry Context
${findings.industryContext}

## Key Topics
${topicList}

## Regulations
${regulationList}

## Learner Insights
${findings.personaInsights}

## Best Practices
${findings.bestPractices.map((bp) => `- ${bp}`).join("\n")}
`.trim();
}

/**
 * Gets a condensed version of the project brief for passing to other agents.
 */
export function getCondensedBrief(brief: ProjectBrief): string {
  let content = `
## Purpose
${brief.purpose}

## Objectives
${brief.objectives.map((o) => `- ${o}`).join("\n")}

## Target Audience
${brief.targetAudience}

## Industry
${brief.industry}

## Scope
In: ${brief.inScope.join(", ")}
Out: ${brief.outOfScope.join(", ")}

## Constraints
${brief.constraints.map((c) => `- ${c}`).join("\n")}`;

  // Add frameworks if present
  if (brief.frameworks && brief.frameworks.length > 0) {
    content += `\n\n## Relevant Frameworks & Standards\n`;
    content += brief.frameworks.map((f) => {
      let line = `- **${f.name}** (${f.type})`;
      if (f.units && f.units.length > 0) {
        line += `\n  Units: ${f.units.join(", ")}`;
      }
      if (f.notes) {
        line += `\n  Notes: ${f.notes}`;
      }
      return line;
    }).join("\n");
  }

  // Add regulations if present
  if (brief.regulations && brief.regulations.length > 0) {
    content += `\n\n## Regulations\n`;
    content += brief.regulations.map((r) => `- ${r}`).join("\n");
  }

  return content.trim();
}

