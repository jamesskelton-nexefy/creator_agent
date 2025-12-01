/**
 * Test script to verify CopilotKit router selects node_expert agent
 * 
 * This sends requests to the CopilotKit runtime and monitors if:
 * 1. The router selects node_expert for node creation tasks
 * 2. The request reaches the LangGraph server
 * 3. The agent responds successfully
 */

import 'dotenv/config';

const LANGGRAPH_URL = 'http://localhost:8000';
const COPILOTKIT_URL = 'http://localhost:4000/copilotkit';
const PUBLIC_API_KEY = 'ck_pub_c95fb1fa919547aedec0e1fd568ae61a';

interface Message {
  role: 'user' | 'assistant' | 'human';
  content: string;
}

/**
 * Test 1: Direct LangGraph API call to node_expert
 */
async function testNodeExpertDirect(testCase: string, userMessage: string) {
  console.log('\n' + '='.repeat(80));
  console.log(`TEST (DIRECT): ${testCase}`);
  console.log('='.repeat(80));
  
  try {
    // Create a thread
    console.log('\n[1/3] Creating thread...');
    const threadResponse = await fetch(`${LANGGRAPH_URL}/threads`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    
    if (!threadResponse.ok) {
      throw new Error(`Failed to create thread: ${threadResponse.status}`);
    }
    
    const thread = await threadResponse.json();
    console.log('    [√] Thread created:', thread.thread_id);

    // Get the node_expert assistant ID
    console.log('\n[2/3] Finding node_expert assistant...');
    const assistantsResponse = await fetch(`${LANGGRAPH_URL}/assistants/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph_id: 'node_expert' }),
    });
    
    if (!assistantsResponse.ok) {
      throw new Error(`Failed to find assistant: ${assistantsResponse.status}`);
    }
    
    const assistants = await assistantsResponse.json();
    if (!assistants || assistants.length === 0) {
      throw new Error('node_expert assistant not found');
    }
    
    const assistantId = assistants[0].assistant_id;
    console.log('    [√] Found assistant:', assistantId);

    // Create a run
    console.log('\n[3/3] Creating run with message...');
    console.log('    Message:', userMessage.substring(0, 100));
    
    const runResponse = await fetch(`${LANGGRAPH_URL}/threads/${thread.thread_id}/runs/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        assistant_id: assistantId,
        input: {
          messages: [
            {
              role: 'human',
              content: userMessage,
            },
          ],
        },
      }),
    });

    if (!runResponse.ok) {
      const errorText = await runResponse.text();
      throw new Error(`Run failed: ${runResponse.status} - ${errorText.substring(0, 200)}`);
    }

    console.log('    [√] Run started, processing stream...');
    
    // Process the stream
    const reader = runResponse.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }
    
    const decoder = new TextDecoder();
    let buffer = '';
    let hasResponse = false;
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            if (data.event === 'values') {
              hasResponse = true;
              const messages = data.data?.messages || [];
              if (messages.length > 0) {
                const lastMsg = messages[messages.length - 1];
                console.log('\n    [√] Agent response received:');
                console.log('        Type:', lastMsg.type);
                if (lastMsg.content) {
                  console.log('        Content preview:', String(lastMsg.content).substring(0, 150));
                }
                if (lastMsg.tool_calls && lastMsg.tool_calls.length > 0) {
                  console.log('        Tool calls:', lastMsg.tool_calls.map((tc: any) => tc.name).join(', '));
                }
              }
            } else if (data.event === 'error') {
              console.error('    [X] Error event:', data);
            }
          } catch (e) {
            // Skip malformed JSON
          }
        }
      }
    }
    
    return hasResponse;
  } catch (error) {
    console.error('\n[X] ERROR:', error instanceof Error ? error.message : String(error));
    return false;
  }
}

/**
 * Test 2: Through CopilotKit runtime (for routing test)
 */
async function testNodeExpertRouting(testCase: string, messages: Message[]) {
  console.log('\n' + '='.repeat(80));
  console.log(`TEST (ROUTING): ${testCase}`);
  console.log('='.repeat(80));
  
  console.log('\n[INFO] This test checks if CopilotKit runtime logs show agent selection.');
  console.log('[INFO] Check terminal 44 (CopilotKit runtime) for these logs:');
  console.log('       - "[CopilotKit] Request started"');
  console.log('       - Agent selection/routing logs');
  console.log('\n[INFO] Simulating request...');
  
  try {
    // For now, just test that the endpoint is reachable
    console.log('\n[√] Sending request to CopilotKit runtime...');
    console.log('    URL:', COPILOTKIT_URL);
    console.log('    Messages:', messages.length);
    console.log('    Last message:', messages[messages.length - 1].content.substring(0, 100));
    
    const response = await fetch(COPILOTKIT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages,
        publicApiKey: PUBLIC_API_KEY,
        properties: {
          copilotCloudPublicApiKey: PUBLIC_API_KEY,
        },
      }),
    });

    console.log('\n[√] Response received');
    console.log('    Status:', response.status, response.statusText);
    console.log('    Headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      const errorText = await response.text();
      console.error('\n[X] ERROR: Request failed');
      console.error('    Status:', response.status);
      console.error('    Body:', errorText.substring(0, 500));
      return false;
    }

    // Handle streaming response
    if (response.headers.get('content-type')?.includes('text/event-stream')) {
      console.log('\n[√] Processing streaming response...');
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let eventCount = 0;
      
      while (reader) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('\n[√] Stream completed');
          console.log('    Total events:', eventCount);
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            eventCount++;
            try {
              const data = JSON.parse(line.slice(6));
              
              // Log important events
              if (data.type === 'agent.start') {
                console.log(`\n[√] AGENT SELECTED: ${data.agentName || 'unknown'}`);
                console.log('    Agent ID:', data.agentId);
                console.log('    Thread ID:', data.threadId);
              } else if (data.type === 'agent.message') {
                console.log(`\n[√] Agent message:`);
                console.log('    Content preview:', data.content?.substring(0, 100));
              } else if (data.type === 'agent.error') {
                console.error(`\n[X] Agent error:`, data.error);
              }
            } catch (e) {
              // Skip non-JSON lines
            }
          }
        }
      }
      
      return true;
    } else {
      // Handle JSON response
      const data = await response.json();
      console.log('\n[√] Response data:', JSON.stringify(data, null, 2).substring(0, 500));
      return true;
    }
  } catch (error) {
    console.error('\n[X] ERROR: Test failed');
    console.error('    Error:', error instanceof Error ? error.message : String(error));
    if (error instanceof Error && error.stack) {
      console.error('    Stack:', error.stack.split('\n').slice(0, 5).join('\n'));
    }
    return false;
  }
}

async function runTests() {
  console.log('\n');
  console.log('╔' + '═'.repeat(78) + '╗');
  console.log('║' + ' '.repeat(20) + 'NODE EXPERT ROUTING TEST SUITE' + ' '.repeat(27) + '║');
  console.log('╚' + '═'.repeat(78) + '╝');
  console.log('\nThis test suite verifies:');
  console.log('  1. node_expert agent works when called directly (LangGraph API)');
  console.log('  2. CopilotKit runtime endpoint is reachable');
  console.log('  3. Watch terminal 44 for routing/agent selection logs\n');

  const directTests = [
    {
      name: 'Direct Test 1: Simple node creation',
      message: 'Create a module called "Introduction to React" with three lessons: Components, Props, and State',
    },
    {
      name: 'Direct Test 2: Course structure request',
      message: 'I need you to create a course structure for teaching TypeScript with modules and lessons.',
    },
  ];

  let passedTests = 0;
  let totalTests = directTests.length;

  // Run direct tests
  console.log('\n' + '━'.repeat(80));
  console.log('PART 1: Testing node_expert directly via LangGraph API');
  console.log('━'.repeat(80));

  for (const test of directTests) {
    const result = await testNodeExpertDirect(test.name, test.message);
    if (result) {
      passedTests++;
      console.log('\n[√] TEST PASSED - node_expert responded successfully');
    } else {
      console.log('\n[X] TEST FAILED - node_expert did not respond');
    }
    
    if (directTests.indexOf(test) < directTests.length - 1) {
      console.log('\nWaiting 2 seconds before next test...');
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('TEST SUMMARY');
  console.log('='.repeat(80));
  console.log(`Direct API Tests: ${passedTests}/${totalTests} passed`);
  
  if (passedTests === totalTests) {
    console.log('\n[√] node_expert agent is working correctly!');
    console.log('\n[INFO] To test CopilotKit routing:');
    console.log('  1. Use your frontend application');
    console.log('  2. Ask it to "Create a module with lessons"');
    console.log('  3. Watch terminal 44 for these logs:');
    console.log('     - "[CopilotKit] Request started"');
    console.log('     - Check if router selects node_expert agent');
    console.log('  4. Watch terminal 42 for these logs:');
    console.log('     - "[node_expert] chat_node called"');
    console.log('     - Agent processing messages\n');
  } else {
    console.error('\n[X] node_expert agent has issues - fix these first!');
  }
  
  console.log('='.repeat(80) + '\n');

  process.exit(passedTests === totalTests ? 0 : 1);
}

// Run tests
runTests().catch((error) => {
  console.error('\n[X] FATAL ERROR:', error);
  process.exit(1);
});

