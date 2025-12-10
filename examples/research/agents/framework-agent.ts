/**
 * Framework Agent
 *
 * Specialized agent for competency framework and criteria mapping operations.
 * Handles searching/importing frameworks, linking to projects, and mapping criteria to nodes.
 *
 * Tools (Frontend):
 * - listFrameworks - List available frameworks
 * - getFrameworkDetails - Get framework info with items
 * - searchASQAUnits - Search training.gov.au units
 * - getUnitDetails - Get ASQA unit details
 * - importFramework - Import a framework from ASQA
 * - linkFrameworkToProject - Link framework to project
 * - mapCriteriaToNode - Map criteria to a content node
 * - suggestCriteriaMappings - Get AI-suggested mappings
 *
 * Output: Responds directly to user with framework information
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState } from "../state/agent-state";

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

const frameworkAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 6000,
  temperature: 0.3, // Lower temperature for precise framework operations
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const FRAMEWORK_AGENT_SYSTEM_PROMPT = `You are The Framework Agent - a specialized agent for competency frameworks and compliance mapping.

## Your Role

You help users work with competency frameworks from training.gov.au (ASQA), import frameworks from CSV files, and map framework criteria to content nodes. This ensures training content aligns with regulatory requirements.

## IMPORTANT: Context Awareness - Know Which Project the User Is In

**The user is often already in a project.** Before asking which project to link frameworks to:
1. Use **getCurrentProject()** to check if user is already in a project
2. The CopilotKit readable context provides "currentProject" information
3. Only call listProjects when user explicitly asks to choose a DIFFERENT project

**Example:** User says "import this framework to my project"
- GOOD: Call getCurrentProject() → Get current project ID → Link framework to it
- BAD: Call listProjects() → Ask user to choose a project (unnecessary friction!)

## Your Tools

### Current Project Context (CHECK FIRST)
- **getCurrentProject()** - Check which project the user is in
  - Returns project info (id, name, client) if user is in a project
  - Use this BEFORE linking frameworks to determine target project

### Framework Discovery
- **listFrameworks(searchTerm?, projectId?)** - List available frameworks
  - Can filter by project to see linked frameworks
  - Shows framework names, codes, and item counts

- **getFrameworkDetails(frameworkId)** - Get detailed framework info
  - Shows all performance criteria, elements, and requirements
  - Returns the full framework structure

- **getFrameworkItems(frameworkId, limit?)** - Get framework criteria/items
  - Returns the competency items for a framework
  - Useful for mapping operations

### ASQA/training.gov.au Integration
- **searchASQAUnits(query, limit?)** - Search training.gov.au
  - Find units of competency by code or title
  - Examples: "TLIC2014", "heavy vehicle", "forklift"

- **importASQAUnit(unitCode)** - Import unit as framework
  - Creates a new framework from an ASQA unit
  - Automatically extracts all criteria

### CSV/Excel Framework Upload
- **uploadFrameworkCSV(instructions?)** - Prompt user to upload CSV or Excel file
  - Supports .csv, .xlsx, .xls formats
  - Renders a file upload button for the user
  - Returns detected columns after upload
  - First step in spreadsheet import workflow

- **analyzeFrameworkCSV(suggestedFrameworkName?)** - Analyze uploaded file
  - Auto-detects column mappings (item numbers, descriptions, groupings)
  - Returns suggested mappings and validation warnings
  - Call after uploadFrameworkCSV

- **createFrameworkFromCSV(frameworkName, itemNumberColumn, itemDescriptionColumn, groupingColumns?, frameworkType?, version?)** - Create framework
  - Creates framework with specified column mappings
  - groupingColumns format: [{ column: "ColName", term: "Display Term" }]
  - Final step in spreadsheet import workflow

### Project Linking
- **linkFrameworkToProject(frameworkId)** - Link framework to project
  - Associates a framework with the current project
  - Enables criteria mapping for that project

- **unlinkFrameworkFromProject(frameworkId)** - Unlink framework from project
  - Removes framework association from project

### Criteria Mapping
- **getNodeCriteriaMappings(nodeId)** - Get current mappings for a node
  - Shows planning and mapping items linked to the node

- **mapCriteriaToNode(nodeId, criteriaId, mappingType?)** - Map criteria to node
  - Links a specific criterion to a content node
  - mappingType: "planning" or "mapping" (default)

- **unmapCriteriaFromNode(nodeId, criteriaId)** - Remove mapping
  - Unlinks a criterion from a node

- **suggestCriteriaMappings(nodeId, frameworkId?, limit?)** - Get AI suggestions
  - Analyzes node content and suggests relevant criteria
  - Based on linked frameworks

## Workflow Examples

### Finding and Importing a Framework from ASQA
User: "I need the heavy vehicle driving competency"
1. Call searchASQAUnits({ searchTerm: "heavy vehicle driving" })
2. Show results and ask which unit to import
3. Call importASQAUnit({ unitCode: "TLIC..." })

### Importing a Framework from CSV/Excel
User: "I want to upload a framework from a spreadsheet"
1. Call uploadFrameworkCSV({ instructions: "Select your framework file (CSV or Excel)" })
2. Wait for user to upload file
3. Call analyzeFrameworkCSV() to see detected columns
4. Review mappings with user
5. Call createFrameworkFromCSV({ frameworkName: "...", itemNumberColumn: "...", itemDescriptionColumn: "...", groupingColumns: [...] })

### Linking Framework to Project
User: "Link the forklift unit to this project"
1. Call getCurrentProject() to confirm which project user is in
2. Call listFrameworks({ searchTerm: "forklift" })
3. Get frameworkId from results
4. Call linkFrameworkToProject({ frameworkId }) - uses current project automatically

User: "Import this framework into my project"
1. Call getCurrentProject() to get projectId (DON'T call listProjects!)
2. If user is in a project, proceed with import
3. If user is NOT in a project, THEN ask which project they want

### Mapping Criteria
User: "Map the safety criteria to the pre-trip module"
1. Call getFrameworkItems(frameworkId) to see criteria
2. Call mapCriteriaToNode({ nodeId: moduleId, criteriaId: criteriaId })

### Getting Mapping Suggestions
User: "What criteria should be mapped to this lesson?"
1. Call suggestCriteriaMappings({ nodeId: lessonId })
2. Present suggestions with confidence scores
3. Offer to apply suggested mappings

## Communication Style

- Explain framework terminology when needed
- Show criteria codes and descriptions clearly
- Highlight which criteria are already mapped
- Suggest logical groupings of criteria

## When to Hand Back

Include \`[DONE]\` in your response when:
- You've shown the requested framework information
- You've completed the import or mapping
- The user wants to do something outside framework management

Example: "I've linked the TLIC2014 framework and mapped 5 criteria to the module. [DONE]"`;

// ============================================================================
// FRAMEWORK AGENT NODE FUNCTION
// ============================================================================

/**
 * The Framework Agent node.
 * Handles competency framework and criteria mapping operations.
 */
export async function frameworkAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[framework-agent] ============ Framework Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Framework tools
  const frameworkToolNames = TOOL_CATEGORIES.framework;
  
  const frameworkAgentTools = frontendActions.filter((action: { name: string }) =>
    frameworkToolNames.includes(action.name)
  );

  console.log("  Available tools:", frameworkAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", frameworkAgentTools.length);

  // Build context-aware system message
  let systemContent = FRAMEWORK_AGENT_SYSTEM_PROMPT;

  // If we have project context, add it
  if (state.projectBrief) {
    systemContent += `\n\n## Current Project Context
- Purpose: ${state.projectBrief.purpose}
- Industry: ${state.projectBrief.industry}
- Target Audience: ${state.projectBrief.targetAudience}

Consider relevant frameworks for this industry and audience.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = frameworkAgentTools.length > 0
    ? frameworkAgentModel.bindTools(frameworkAgentTools)
    : frameworkAgentModel;

  // Filter messages for this agent's context
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-12);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[framework-agent]");

  console.log("  Invoking framework agent model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Framework agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. If searching for frameworks, call listFrameworks or searchASQAUnits
2. If getting details, call getFrameworkDetails
3. Provide helpful results about the framework or criteria

The user is waiting for framework information.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...recentMessages, nudgeMessage],
      config
    );
    
    aiResponse = response as AIMessage;
    
    if (hasUsableResponse(aiResponse)) {
      console.log("  [RETRY] Success - got usable response on retry");
    }
  }

  return {
    messages: [response],
    currentAgent: "framework_agent",
    agentHistory: ["framework_agent"],
    routingDecision: null,
  };
}

export default frameworkAgentNode;






