# OmniChunkManager Architecture

## Overview

The OmniChunkManager provides asynchronous chunk loading and saving for multi-stage model execution. This document describes the architecture after refactoring to use a clean base class design.

## Class Diagram

```mermaid
classDiagram
    %% --- Schedulers ---
    class OmniARScheduler {
        +schedule()
        +update_from_output()
        -chunk_manager: OmniChunkManager
    }
    
    class OmniGenerationScheduler {
        +schedule()
        +update_from_output()
        -chunk_manager: OmniChunkManager
    }

    %% --- Base Chunk Manager ---
    class BaseOmniChunkManager {
        <<abstract>>
        <<vllm_omni.distributed.omni_connectors.base_chunk_manager>>
        
        #connector: OmniConnectorBase
        #_pending_load_reqs: dict
        #_finished_load_reqs: set
        #_pending_save_reqs: dict
        #_finished_save_reqs: set
        #recv_thread: Thread
        #save_thread: Thread
        
        %% Public Interface
        +get_finished_load_requests() : set
        +get_finished_save_requests() : set
        +request_chunk(request)*
        +submit_chunk(output, request, func)*
        
        %% Internal Methods
        #_recv_loop()
        #_save_loop()
        #_process_load_request(req_id)*
        #_get_next_save_task() : dict*
        #_process_save_task(task)*
        #_mark_load_finished(req_id)
        #_mark_save_finished(req_id)
    }

    %% --- Chunk Manager Implementation ---
    class OmniChunkManager {
        <<vllm_omni.distributed.omni_connectors.adapter>>
        
        %% Scheduler Helper Methods
        +request_chunk(request)
        +submit_chunk(output, request, func)
        +get_finished_load_requests() : set
        
        %% Legacy Compatibility
        +get_chunk(request)
        +put_chunk(output, request, func)
        +get_finished() : set
        
        %% Internal Implementation
        #_process_load_request(req_id)
        #_get_next_save_task() : dict
        #_process_save_task(task)
    }

    %% --- Future KV Transfer Manager ---
    class OmniKVTransferManager {
        <<future extension>>
        <<vllm_omni.distributed.omni_connectors.kv_transfer_manager>>
        
        %% KV Cache Transfer Methods
        +handle_finished_requests_kv_transfer()
        +receive_kv_cache_for_request()
        +apply_kv_cache_to_request()
        
        %% Internal Methods
        #_extract_kv_cache()
        #_transfer_kv_cache()
        #_transfer_with_retry()
    }

    %% --- Connectors (Transport Layer) ---
    class OmniConnectorBase {
        <<abstract>>
        <<vllm_omni.distributed.omni_connectors.connectors.base>>
        +put(from_stage, to_stage, key, data)*
        +get(from_stage, to_stage, key)*
        +serialize_obj(obj)*
        +deserialize_obj(data)*
    }
    
    class SharedMemoryConnector {
        <<vllm_omni.distributed.omni_connectors.connectors.shm_connector>>
        +put(from_stage, to_stage, key, data)
        +get(from_stage, to_stage, key)
    }

    %% --- Relationships ---
    OmniARScheduler --> OmniChunkManager : uses
    OmniGenerationScheduler --> OmniChunkManager : uses
    
    OmniChunkManager --|> BaseOmniChunkManager : extends
    OmniKVTransferManager -.-> BaseOmniChunkManager : future extension
    
    BaseOmniChunkManager --> OmniConnectorBase : uses
    SharedMemoryConnector --|> OmniConnectorBase : implements
```

## Architecture Design Principles

### 1. Separation of Concerns

- **Schedulers**: Focus on scheduling logic and request lifecycle management
- **Chunk Manager**: Handles asynchronous chunk loading/saving operations
- **Connectors**: Provide transport layer abstraction (SHM, Mooncake, Yuanrong)

### 2. Base Class Design

The `BaseOmniChunkManager` provides:

- **Request Queue Management**: Tracks pending and finished load/save requests
- **Thread Management**: Background threads for async I/O operations
- **Lifecycle Management**: Start/stop mechanisms for clean shutdown
- **Abstract Interface**: Defines contract for subclass implementations

### 3. Delegation Pattern

Schedulers delegate chunk operations to the chunk manager:

```python
# In Scheduler.__init__
self.chunk_manager = OmniChunkManager(connector)

# In Scheduler.schedule()
self.chunk_manager.request_chunk(request)
finished = self.chunk_manager.get_finished_load_requests()

# In Scheduler.update_from_output()
self.chunk_manager.submit_chunk(output, request, process_func)
```

### 4. Connector Abstraction

The chunk manager uses connectors for actual data transfer:

```python
# In OmniChunkManager._process_load_request()
result = self.connector.get(from_stage, to_stage, key)

# In OmniChunkManager._process_save_task()
success, size, metadata = self.connector.put(from_stage, to_stage, key, data)
```

## Component Responsibilities

### BaseOmniChunkManager

**Purpose**: Provide common infrastructure for async chunk/cache operations

**Responsibilities**:
- Manage request queues (pending/finished)
- Run background threads for I/O
- Define abstract interface for subclasses
- Provide helper methods for state management

**Key Methods**:
- `get_finished_load_requests()`: Get completed load requests
- `request_chunk(request)`: Enqueue a load request
- `submit_chunk(output, request, func)`: Enqueue a save request

### OmniChunkManager

**Purpose**: Implement chunk management for multi-stage models

**Responsibilities**:
- Process chunk load requests from previous stage
- Process chunk save requests to next stage
- Handle stage-specific logic (e.g., qwen3_omni)
- Maintain backward compatibility

**Key Methods**:
- `_process_load_request(req_id)`: Fetch and process a chunk
- `_process_save_task(task)`: Send a chunk via connector
- Legacy methods for backward compatibility

### OmniKVTransferManager (Existing)

**Purpose**: Handle KV cache transfer between stages

**Responsibilities**:
- Extract KV cache from GPU blocks
- Transfer KV cache via connector
- Receive and apply KV cache
- Currently operates independently, can be unified with base class in future

## Usage Examples

### Scheduler Integration

```python
# Initialize chunk manager
connector = OmniConnectorFactory.create_connector(connector_spec)
self.chunk_manager = OmniChunkManager(connector)

# Request chunk loading
def schedule(self):
    for request in self.waiting:
        self.chunk_manager.request_chunk(request)
    
    finished_req_ids = self.chunk_manager.get_finished_load_requests()
    # ... process finished requests

# Submit chunk for saving
def update_from_output(self, scheduler_output, model_output):
    for request in finished_requests:
        self.chunk_manager.submit_chunk(
            pooler_output, 
            request, 
            custom_process_func
        )
```

### Custom Processing Function

```python
def custom_process_next_stage_input(pooling_output, request):
    """Extract and format data for next stage."""
    return {
        "thinker_embeddings": pooling_output.get("embeddings"),
        "thinker_hidden_states": pooling_output.get("hidden_states"),
        "finished": request.status == RequestStatus.FINISHED_STOPPED,
    }
```

## Future Extensions

### 1. Unified Base Class for KV Transfer

The existing `OmniKVTransferManager` can be refactored to:
- Extend `BaseOmniChunkManager` for common infrastructure
- Implement specific methods for KV cache operations
- Share thread management and connector abstraction

### 2. Configurable Processing Pipeline

```python
class ConfigurableChunkManager(BaseOmniChunkManager):
    def __init__(self, connector, pipeline):
        super().__init__(connector)
        self.processing_pipeline = pipeline
    
    def _process_load_request(self, req_id):
        data = self.connector.get(...)
        for processor in self.processing_pipeline:
            data = processor.transform(data)
        # ... apply to request
```

### 3. Metrics and Monitoring

```python
class InstrumentedChunkManager(OmniChunkManager):
    def _process_load_request(self, req_id):
        start = time.time()
        super()._process_load_request(req_id)
        self.metrics.record_load_latency(time.time() - start)
```

## Performance Considerations

1. **Non-blocking I/O**: Background threads poll without blocking scheduler
2. **Lock Minimization**: Fine-grained locking only on shared state
3. **Batch Processing**: Save queue allows batching multiple chunks
4. **Early Return**: Quick checks before expensive operations

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test scheduler + chunk manager + connector
3. **End-to-End Tests**: Full multi-stage pipeline tests
4. **Performance Tests**: Measure latency and throughput

## Migration Guide

### From Legacy Interface

Old code:
```python
self.chunk_manager.get_chunk(request)
finished = self.chunk_manager.get_finished()
self.chunk_manager.put_chunk(output, request, func)
```

New code (recommended):
```python
self.chunk_manager.request_chunk(request)
finished = self.chunk_manager.get_finished_load_requests()
self.chunk_manager.submit_chunk(output, request, func)
```

Note: Legacy methods still work for backward compatibility.
