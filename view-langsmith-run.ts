/**
 * LangSmith Run Viewer
 * 
 * Fetches and displays details of a specific LangSmith run or thread.
 * Usage: 
 *   npx tsx view-langsmith-run.ts <run-id>
 *   npx tsx view-langsmith-run.ts --thread <thread-id>
 */

import 'dotenv/config';
import { Client } from 'langsmith';

const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY;

if (!LANGSMITH_API_KEY) {
  console.error('[X] LANGSMITH_API_KEY not found in environment');
  process.exit(1);
}

const client = new Client({
  apiKey: LANGSMITH_API_KEY,
});

async function viewRun(runId: string, detailed = false) {
  console.log('\n' + '='.repeat(80));
  console.log('LANGSMITH RUN VIEWER');
  console.log('='.repeat(80));
  console.log(`\nRun ID: ${runId}`);

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
    
    if (detailed) {
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
    const langsmithUrl = `https://smith.langchain.com/o/10cdf0e9-9d6a-4cd9-a4ae-bc99bdf1f60f/projects/p/${run.session_id}?peek=${runId}`;
    
    console.log('\n' + '='.repeat(80));
    console.log('VIEW IN LANGSMITH');
    console.log('='.repeat(80));
    console.log(`\n${langsmithUrl}\n`);
    
    return run;
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

async function viewThread(threadId: string, projectOverride?: string) {
  console.log('\n' + '='.repeat(80));
  console.log('LANGSMITH THREAD VIEWER');
  console.log('='.repeat(80));
  console.log(`\nThread ID: ${threadId}`);
  console.log('\n[i] Fetching runs for thread...\n');

  try {
    const projectName = projectOverride || process.env.LANGSMITH_PROJECT || 'default';
    console.log(`[i] Searching in project: ${projectName}`);
    
    // Query runs that have this thread_id in their metadata
    // LangGraph stores thread_id in the configurable metadata
    const runs: any[] = [];
    
    // Search all recent runs in the project for the thread_id
    // The listRuns async iterator auto-paginates
    console.log('[i] Scanning runs (this may take a moment)...');
    let scannedCount = 0;
    const seenThreadIds = new Set<string>();
    const MAX_SCAN = 2000; // Max runs to scan
    
    for await (const run of client.listRuns({
      projectName,
    })) {
      scannedCount++;
      
      const runThreadId = run.extra?.metadata?.thread_id;
      if (runThreadId) {
        seenThreadIds.add(runThreadId);
      }
      
      // Check if thread_id is in run metadata
      if (runThreadId === threadId) {
        runs.push(run);
      }
      
      // Progress update
      if (scannedCount % 500 === 0) {
        console.log(`[i] Scanned ${scannedCount} runs, found ${runs.length} matching...`);
      }
      
      // Stop if we've scanned enough
      if (scannedCount >= MAX_SCAN) {
        console.log(`[i] Reached scan limit of ${MAX_SCAN}`);
        break;
      }
    }
    
    console.log(`\n[i] Scanned ${scannedCount} runs, found ${runs.length} matching thread_id`);
    console.log(`[i] Unique thread_ids seen: ${seenThreadIds.size}`);
    
    // Show recent thread_ids if not found
    if (runs.length === 0 && seenThreadIds.size > 0) {
      console.log('\n[i] Recent thread_ids found (sample):');
      const threadArray = Array.from(seenThreadIds).slice(0, 10);
      for (const tid of threadArray) {
        console.log(`    - ${tid}`);
      }
    }

    if (runs.length === 0) {
      console.log('\n[X] No runs found for this thread ID.');
      console.log('\n[i] Tips:');
      console.log('    - Verify the thread ID is correct');
      console.log('    - Check that runs were traced to LangSmith');
      console.log('    - The thread may have been created recently and not yet indexed');
      return;
    }

    // Sort runs by start time
    runs.sort((a, b) => new Date(a.start_time).getTime() - new Date(b.start_time).getTime());

    console.log('='.repeat(80));
    console.log(`THREAD RUNS (${runs.length} found)`);
    console.log('='.repeat(80));

    // Calculate total stats
    let totalLatency = 0;
    let totalTokens = 0;
    let errorCount = 0;
    const toolCallCounts: Record<string, number> = {};

    for (const run of runs) {
      if (run.latency_ms) totalLatency += run.latency_ms;
      if (run.total_tokens) totalTokens += run.total_tokens;
      if (run.error) errorCount++;
    }

    console.log('\n--- THREAD SUMMARY ---');
    console.log(`Total Runs:     ${runs.length}`);
    console.log(`Total Latency:  ${totalLatency}ms (${(totalLatency / 1000).toFixed(2)}s)`);
    console.log(`Total Tokens:   ${totalTokens}`);
    console.log(`Errors:         ${errorCount}`);

    console.log('\n--- RUNS TIMELINE ---\n');

    for (let i = 0; i < runs.length; i++) {
      const run = runs[i];
      const startTime = new Date(run.start_time);
      const statusIcon = run.error ? 'X' : run.status === 'success' ? 'v' : '?';
      
      console.log(`[${i + 1}] ${statusIcon} ${run.name} (${run.run_type})`);
      console.log(`    ID:       ${run.id}`);
      console.log(`    Status:   ${run.status}`);
      console.log(`    Time:     ${startTime.toISOString()}`);
      console.log(`    Latency:  ${run.latency_ms ? `${run.latency_ms}ms` : 'N/A'}`);
      console.log(`    Tokens:   ${run.total_tokens || 'N/A'}`);
      
      if (run.error) {
        console.log(`    ERROR:    ${typeof run.error === 'string' ? run.error.substring(0, 200) : JSON.stringify(run.error).substring(0, 200)}...`);
      }

      // Check for tool calls in inputs/outputs
      if (run.inputs?.tool_calls) {
        console.log(`    Tool Calls: ${JSON.stringify(run.inputs.tool_calls.map((t: any) => t.name || t.type))}`);
      }
      
      console.log('');
    }

    // Find slow runs
    const slowRuns = runs.filter(r => r.latency_ms > 10000).sort((a, b) => b.latency_ms - a.latency_ms);
    if (slowRuns.length > 0) {
      console.log('\n--- SLOW RUNS (>10s) ---\n');
      for (const run of slowRuns) {
        console.log(`  ${run.name}: ${run.latency_ms}ms (${(run.latency_ms / 1000).toFixed(2)}s)`);
        console.log(`    ID: ${run.id}`);
      }
    }

    // Find runs with errors
    const errorRuns = runs.filter(r => r.error);
    if (errorRuns.length > 0) {
      console.log('\n--- ERROR RUNS ---\n');
      for (const run of errorRuns) {
        console.log(`  ${run.name}`);
        console.log(`    ID: ${run.id}`);
        console.log(`    Error: ${typeof run.error === 'string' ? run.error.substring(0, 300) : JSON.stringify(run.error).substring(0, 300)}`);
        console.log('');
      }
    }

    // Detailed analysis prompt
    console.log('\n' + '='.repeat(80));
    console.log('DETAILED ANALYSIS');
    console.log('='.repeat(80));
    console.log('\nTo view a specific run in detail, run:');
    console.log(`  npx tsx view-langsmith-run.ts <run-id>`);
    console.log('\nTop-level runs (no parent):');
    const topLevelRuns = runs.filter(r => !r.parent_run_id);
    for (const run of topLevelRuns) {
      console.log(`  - ${run.id} (${run.name})`);
    }

    // Generate LangSmith project URL
    console.log('\n' + '='.repeat(80));
    console.log('VIEW IN LANGSMITH');
    console.log('='.repeat(80));
    console.log(`\nProject: ${projectName}`);
    if (runs[0]?.session_id) {
      console.log(`URL: https://smith.langchain.com/o/10cdf0e9-9d6a-4cd9-a4ae-bc99bdf1f60f/projects/p/${runs[0].session_id}\n`);
    }

  } catch (error) {
    console.error('\n[X] Error fetching thread runs:', error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
let mode = 'run';
let targetId = '';
let projectOverride: string | undefined;

// Parse args
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--thread' || arg === '-t') {
    mode = 'thread';
    targetId = args[++i] || '';
  } else if (arg === '--project' || arg === '-p') {
    projectOverride = args[++i];
  } else if (arg === '--help' || arg === '-h') {
    console.log(`
LangSmith Run Viewer

Usage:
  npx tsx view-langsmith-run.ts <run-id>                     View a specific run
  npx tsx view-langsmith-run.ts --thread <thread-id>         View all runs for a thread
  npx tsx view-langsmith-run.ts -t <thread-id> -p <project>  Specify project name

Options:
  --thread, -t    View runs by thread ID instead of run ID
  --project, -p   Override the LANGSMITH_PROJECT env var
  --help, -h      Show this help message

Examples:
  npx tsx view-langsmith-run.ts -t abc123 -p blueprint-agents
  npx tsx view-langsmith-run.ts run-id-here
`);
    process.exit(0);
  } else if (!targetId) {
    targetId = arg;
  }
}

if (!targetId) {
  console.error('[X] No ID provided. Use --help for usage.');
  process.exit(1);
}

if (mode === 'thread') {
  viewThread(targetId, projectOverride);
} else {
  viewRun(targetId, true);
}

