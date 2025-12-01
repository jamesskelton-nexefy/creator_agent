import "dotenv/config";
import { z } from "zod";
import { tool } from "langchain";
import { TavilySearch } from "@langchain/tavily";
import { ChatAnthropic } from "@langchain/anthropic";

import {
  createDeepAgent,
  createCopilotKitMiddleware,
  type SubAgent,
} from "../../src/index.js";

type Topic = "general" | "news" | "finance";

// Search tool to use to do research
const internetSearch = tool(
  async ({
    query,
    maxResults = 5,
    topic = "general" as Topic,
    includeRawContent = false,
  }: {
    query: string;
    maxResults?: number;
    topic?: Topic;
    includeRawContent?: boolean;
  }) => {
    /**
     * Run a web search
     */

    // Note: You'll need to install and import tavily-js or similar package
    // For now, this is a placeholder that shows the structure
    const tavilySearch = new TavilySearch({
      maxResults,
      tavilyApiKey: process.env.TAVILY_API_KEY,
      includeRawContent,
      topic,
    });
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore - Type instantiation is excessively deep and possibly infinite.
    const tavilyResponse = await tavilySearch._call({ query });

    return tavilyResponse;
  },
  {
    name: "internet_search",
    description: "Run a web search",
    schema: z.object({
      query: z.string().describe("The search query"),
      maxResults: z
        .number()
        .optional()
        .default(5)
        .describe("Maximum number of results to return"),
      topic: z
        .enum(["general", "news", "finance"])
        .optional()
        .default("general")
        .describe("Search topic category"),
      includeRawContent: z
        .boolean()
        .optional()
        .default(false)
        .describe("Whether to include raw content"),
    }),
  },
);

const subResearchPrompt = `You are a dedicated researcher. Your job is to conduct research based on the users questions.

CConduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!`;

const researchSubAgent: SubAgent = {
  name: "research-agent",
  description:
    "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
  systemPrompt: subResearchPrompt,
  tools: [internetSearch],
};

const subCritiquePrompt = `You are a dedicated editor. You are being tasked to critique a report.

You can find the report at \`final_report.md\`.

You can find the question/topic for this report at \`question.txt\`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the \`final_report.md\` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
`;

const critiqueSubAgent: SubAgent = {
  name: "critique-agent",
  description:
    "Used to critique the final report. Give this agent some infomration about how you want it to critique the report.",
  systemPrompt: subCritiquePrompt,
};

// Node management subagent system prompt - now focuses on planning/designing node structures
const nodeAgentPrompt = `You are a content structure specialist for BlueprintCMS. Your role is to help users PLAN and DESIGN content node structures.

## Your Role

You help users think through and design content hierarchies. You do NOT have access to frontend tools.
Instead, you output structured JSON that the frontend will use to create nodes.

## Hierarchy Understanding

Each project has hierarchy levels (typically 2-6):
- **Level 1** is the project root (auto-created, never create manually)
- **Level 2** is where user content starts (called "Section", "Module", or "Chapter" depending on project)
- **Levels 3-6** are for progressively more detailed content

**Coding Types:**
- \`numeric\`: 1, 2, 3... 
- \`alpha\`: A, B, C...
- \`title\`: Uses node title

## How to Design Node Structures

When asked to create nodes or design content structures:

1. **Research the topic** if needed (use internet_search)
2. **Plan the hierarchy** - think about modules, lessons, topics
3. **Output structured JSON** in this format:

\`\`\`json
{
  "plannedNodes": [
    { "id": "1", "title": "Module Name", "type": "module", "content": "Description..." },
    { "id": "2", "title": "Lesson Name", "type": "lesson", "parentId": "1", "content": "..." },
    { "id": "3", "title": "Topic Name", "type": "topic", "parentId": "2", "content": "..." }
  ]
}
\`\`\`

The frontend will automatically create these nodes when it parses your response.

## Key Rules

1. **NEVER reference frontend tools** - you don't have access to them
2. **Always output plannedNodes JSON** for node creation requests
3. **Use parentId** to establish hierarchy relationships
4. **Start at level 2** - never try to create level 1 (project root)
5. **Be thorough** - include content descriptions for each node
`;

const nodeSubAgent: SubAgent = {
  name: "node-agent",
  description:
    "Specialized agent for designing and planning content node structures. Use this to think through how to organize content into modules, lessons, and topics. This agent outputs structured JSON that the frontend uses to create nodes.",
  systemPrompt: nodeAgentPrompt,
};

// Prompt prefix to steer the agent to be an expert researcher
const researchInstructions = `You are an expert researcher and helpful assistant. Your primary job is to conduct thorough research and write polished reports. You can also help users design content structures.

## Research Tasks

For research tasks, the first thing you should do is to write the original user question to \`question.txt\` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to \`final_report.md\`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the \`final_report.md\`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>

CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## \`internet_search\`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.

## When to Use Which Tools

- **User asks a research question or wants external information** → Use internet_search or delegate to research-agent subagent
- **User wants to critique a report** → Use the critique-agent subagent
- **User asks to create nodes, content structures, or project hierarchies** → Research and plan, then output structured JSON (see below)

## Creating Nodes and Content Structures

When asked to create nodes, design course structures, or build content hierarchies:
1. Research the topic thoroughly using internet_search if needed
2. Plan the structure carefully (modules, lessons, topics, etc.)
3. Output your planned nodes as structured JSON in your response

**IMPORTANT**: Do NOT call frontend tools directly. Instead, include your node plan in this format:

\`\`\`json
{
  "plannedNodes": [
    { "id": "1", "title": "Module Name", "type": "module", "content": "Description of this module..." },
    { "id": "2", "title": "Lesson Name", "type": "lesson", "parentId": "1", "content": "Lesson content..." },
    { "id": "3", "title": "Topic Name", "type": "topic", "parentId": "2", "content": "Topic details..." }
  ]
}
\`\`\`

Use hierarchical IDs and parentId to establish the structure. The frontend will parse this JSON and create the nodes automatically.

## Available Subagents

- **research-agent**: For in-depth research on topics. Give one topic at a time.
- **critique-agent**: For reviewing and critiquing reports in final_report.md.
- **node-agent**: For understanding hierarchy structures and node templates.

Always respond helpfully with well-researched, structured content.
`;

// Create the agent with CopilotKit middleware for frontend tool support
export const agent = createDeepAgent({
  model: new ChatAnthropic({
    model: "claude-sonnet-4-20250514",
    temperature: 0,
  }),

  tools: [internetSearch],
  systemPrompt: researchInstructions,
  subagents: [critiqueSubAgent, researchSubAgent, nodeSubAgent],

  // Add CopilotKit middleware for context injection and shared state
  // Frontend tools are handled via shared state pattern (see docs: https://docs.copilotkit.ai/langgraph/shared-state)
  middleware: [
    createCopilotKitMiddleware({
      includeContextInPrompt: true,
    }),
  ],
});

// Invoke the agent with recursion limit
import { HumanMessage } from "@langchain/core/messages";

async function main() {
  const result = await agent.invoke(
    {
      messages: [new HumanMessage("what is langgraph?")],
    },
    { recursionLimit: 100 } // Set recursion limit (default is 25)
  );

  console.log("Finished!");
  console.log(
    `\n\nAgent ToDo List:\n${result.todos.map((todo: { content: string; status: string }) => ` - ${todo.content} (${todo.status})`).join("\n")}`
  );
  console.log(
    `\n\nAgent Files:\n${Object.entries(result.files)
      .map(([key, value]) => ` - ${key}: ${value}`)
      .join("\n")}`
  );
}

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
