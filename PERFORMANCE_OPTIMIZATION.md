# Performance Optimization Guide for TalentJobMatch API

## Overview
This document explains the performance issues with the `/api/v1/match/candidate` endpoint and the optimizations implemented to improve response time.

## Root Causes of Slow Performance

### 1. Inefficient Vector Database Queries
- The original implementation was fetching ALL documents from the vector store every time
- This became increasingly slow as the database grew larger
- Memory usage increased significantly with large document collections

### 2. Multiple LLM Calls Per Request
- Multi-query generation via `get_multi_query_variants`
- Individual explanation generation for each candidate via `generate_explanations`
- Faithfulness and relevancy evaluation for each candidate via `evaluate_candidate`
- Each LLM call adds significant latency (typically 1-5 seconds per call)

### 3. Sequential Processing
- All pipeline steps were executed sequentially without any parallelization
- No caching mechanisms for expensive operations

## Implemented Optimizations

### 1. Optimized Vector Store Usage
- **File**: `app/vector_store.py`
- Added optimized model parameters to improve embedding speed
- Specified CPU device and normalization settings for faster computation

### 2. Improved Search Algorithm
- **File**: `app/search.py`
- Optimized the hybrid search algorithm to avoid loading all documents into memory
- Used vector store's native retrieval to reduce memory footprint
- Implemented better deduplication strategy

### 3. Performance Monitoring
- **File**: `app/performance_monitor.py`
- Added timing decorators to track execution time of key functions
- Created performance monitoring utility to identify bottlenecks

### 4. Endpoint-Level Monitoring
- **File**: `app/server.py`
- Added performance monitoring to the main endpoint
- Track total execution time and individual pipeline performance

## Additional Recommendations

### Short-term Solutions:
1. **Reduce candidate count**: Lower the number of candidates processed per request
2. **Implement basic caching**: Cache results for identical job descriptions
3. **Optimize LLM prompts**: Make prompts more efficient to reduce token usage

### Long-term Solutions:
1. **Parallel processing**: Process candidates in parallel where possible
2. **Asynchronous operations**: Use async/await for I/O operations
3. **Database indexing**: Improve vector database indexing strategy
4. **Model optimization**: Consider smaller/faster embedding models for initial filtering
5. **Result caching**: Implement Redis or similar for caching common queries

## Performance Testing

To test the performance improvements, run:

```bash
python performance_test.py
```

This will execute multiple test requests and provide a detailed performance report.

## Expected Improvements

With these optimizations, you should see:
- Reduced memory usage during vector searches
- Faster response times (especially with larger databases)
- Better identification of bottlenecks through performance monitoring
- Improved scalability as the database grows

## Monitoring Results

The performance monitor will output timing information like:
```
⏱️  match_candidates took 15.23 seconds
📊 PERFORMANCE REPORT
--------------------------------------------------
hiring_pipeline_execution: 14.50s avg over 1 calls
match_candidates_total: 15.23s avg over 1 calls
```

Use this information to identify which parts of the pipeline need further optimization.