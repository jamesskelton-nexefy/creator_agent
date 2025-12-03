/**
 * Shared State Definitions for Multi-Agent Orchestrator
 *
 * Defines the state annotation that flows between all agents in the system.
 * Each agent reads from and writes to specific keys in this shared state.
 */

import { Annotation } from "@langchain/langgraph";
import { CopilotKitStateAnnotation } from "@copilotkit/sdk-js/langgraph";
import { BaseMessage } from "@langchain/core/messages";

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
 */
export type AgentType = 
  | "orchestrator"
  | "strategist" 
  | "researcher" 
  | "architect" 
  | "writer" 
  | "visual_designer";

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
// STATE ANNOTATION - Shared state for all agents
// ============================================================================

/**
 * Shared state annotation that all agents in the system use.
 * Extends CopilotKitStateAnnotation to inherit frontend tool access.
 */
export const OrchestratorStateAnnotation = Annotation.Root({
  // Inherit CopilotKit state (messages, actions, context)
  ...CopilotKitStateAnnotation.spec,

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

  /** Course structure from the Architect */
  courseStructure: Annotation<CourseStructure | null>({
    reducer: (existing, update) => update ?? existing,
    default: () => null,
  }),

  /** Visual design from the Visual Designer */
  visualDesign: Annotation<VisualDesign | null>({
    reducer: (existing, update) => update ?? existing,
    default: () => null,
  }),

  /** Content outputs from the Writer (accumulates) */
  writtenContent: Annotation<ContentOutput[]>({
    reducer: (existing, update) => {
      // Merge new content with existing, avoiding duplicates by nodeId
      const merged = [...(existing || [])];
      for (const content of update || []) {
        if (!merged.some((c) => c.nodeId === content.nodeId)) {
          merged.push(content);
        }
      }
      return merged;
    },
    default: () => [],
  }),

  // ---- Execution Tracking ----

  /** Track created nodes to prevent duplicates */
  createdNodes: Annotation<CreatedNode[]>({
    reducer: (existing, update) => {
      const merged = [...(existing || [])];
      for (const node of update || []) {
        if (!merged.some((n) => n.nodeId === node.nodeId)) {
          merged.push(node);
        }
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
   * Tracks when the agent is waiting for async user action (e.g., file upload).
   * Prevents premature graph re-entry during async operations.
   * Set to a string describing the action (e.g., "document_upload"), cleared with null.
   */
  awaitingUserAction: Annotation<string | null>({
    reducer: (existing, update) => update === undefined ? existing : update,
    default: () => null,
  }),

  /** Which agent is currently active */
  currentAgent: Annotation<AgentType>({
    reducer: (existing, update) => update ?? existing,
    default: () => "orchestrator",
  }),

  /** Tracks which agents have been invoked this session */
  agentHistory: Annotation<AgentType[]>({
    reducer: (existing, update) => [...(existing || []), ...(update || [])],
    default: () => [],
  }),

  /** Error state if an agent fails */
  lastError: Annotation<string | null>({
    reducer: (existing, update) => update,
    default: () => null,
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
  return `
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
${brief.constraints.map((c) => `- ${c}`).join("\n")}
`.trim();
}

