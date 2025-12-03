/**
 * LangSmith Run Viewer
 * 
 * Fetches and displays details of a specific LangSmith run.
 * Usage: npx tsx view-langsmith-run.ts <run-id>
 */

import 'dotenv/config';
import { Client } from 'langsmith';

const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY;

if (!LANGSMITH_API_KEY) {
  console.error('[X] LANGSMITH_API_KEY not found in environment');
  process.exit(1);
}

async function viewRun(runId: string) {
  console.log('\n' + '='.repeat(80));
  console.log('LANGSMITH RUN VIEWER');
  console.log('='.repeat(80));
  console.log(`\nRun ID: ${runId}`);
  
  const client = new Client({
    apiKey: LANGSMITH_API_KEY,
  });

  try {
    console.log('\n[i] Fetching run details...\n');
    
    const run = await client.readRun(runId);
    
    console.log('=' .repeat(80));
    console.log('RUN DETAILS');
    console.log('='.repeat(80));
    
    console.log(`\nName:          ${run.name}`);
    console.log(`Run Type:      ${run.run_type}`);
    console.log(`Status:        ${run.status}`);
    console.log(`Start Time:    ${run.start_time}`);
    console.log(`End Time:      ${run.end_time || 'N/A'}`);
    console.log(`Latency:       ${run.latency_ms ? `${run.latency_ms}ms` : 'N/A'}`);
    console.log(`Total Tokens:  ${run.total_tokens || 'N/A'}`);
    console.log(`Prompt Tokens: ${run.prompt_tokens || 'N/A'}`);
    console.log(`Completion:    ${run.completion_tokens || 'N/A'}`);
    console.log(`Session ID:    ${run.session_id || 'N/A'}`);
    console.log(`Parent Run:    ${run.parent_run_id || 'N/A'}`);
    
    if (run.error) {
      console.log('\n' + '-'.repeat(80));
      console.log('ERROR (FULL)');
      console.log('-'.repeat(80));
      // Print the FULL error without truncation
      console.log(typeof run.error === 'string' ? run.error : JSON.stringify(run.error, null, 2));
    }
    
    console.log('\n' + '-'.repeat(80));
    console.log('INPUTS');
    console.log('-'.repeat(80));
    console.log(JSON.stringify(run.inputs, null, 2));
    
    console.log('\n' + '-'.repeat(80));
    console.log('OUTPUTS');
    console.log('-'.repeat(80));
    console.log(JSON.stringify(run.outputs, null, 2));
    
    if (run.extra) {
      console.log('\n' + '-'.repeat(80));
      console.log('EXTRA METADATA');
      console.log('-'.repeat(80));
      console.log(JSON.stringify(run.extra, null, 2));
    }

    // Fetch child runs if this is a parent
    console.log('\n' + '-'.repeat(80));
    console.log('CHILD RUNS');
    console.log('-'.repeat(80));
    
    const childRuns: any[] = [];
    for await (const childRun of client.listRuns({
      parentRunId: runId,
    })) {
      childRuns.push(childRun);
    }
    
    if (childRuns.length > 0) {
      console.log(`\nFound ${childRuns.length} child runs:\n`);
      for (const child of childRuns) {
        console.log(`  - ${child.name} (${child.run_type})`);
        console.log(`    ID: ${child.id}`);
        console.log(`    Status: ${child.status}`);
        console.log(`    Latency: ${child.latency_ms ? `${child.latency_ms}ms` : 'N/A'}`);
        if (child.error) {
          console.log(`    Error: ${child.error.substring(0, 100)}...`);
        }
        console.log('');
      }
    } else {
      console.log('\nNo child runs found.');
    }

    // Generate LangSmith URL
    const projectName = process.env.LANGSMITH_PROJECT || 'default';
    const langsmithUrl = `https://smith.langchain.com/o/10cdf0e9-9d6a-4cd9-a4ae-bc99bdf1f60f/projects/p/${run.session_id}?peek=${runId}`;
    
    console.log('\n' + '='.repeat(80));
    console.log('VIEW IN LANGSMITH');
    console.log('='.repeat(80));
    console.log(`\n${langsmithUrl}\n`);
    
  } catch (error) {
    console.error('\n[X] Error fetching run:', error instanceof Error ? error.message : String(error));
    
    if (error instanceof Error && error.message.includes('404')) {
      console.error('\n[i] Run not found. Check:');
      console.error('    - The run ID is correct');
      console.error('    - The run exists in your LangSmith project');
      console.error('    - Your API key has access to this run');
    }
    
    process.exit(1);
  }
}

// Get run ID from command line args or use default
const runId = process.argv[2] || 'c57e3293-26fc-4e18-8204-aacb9579e3bd';

viewRun(runId);

