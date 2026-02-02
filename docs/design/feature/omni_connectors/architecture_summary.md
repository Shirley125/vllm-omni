# OmniConnector Architecture Summary

## Updated Class Diagram

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

    %% --- Base Chunk Manager (NEW) ---
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
        
        %% Scheduler Helper Methods
        +get_finished_load_requests() : set
        +get_finished_save_requests() : set
        +request_chunk(request)*
        +submit_chunk(output, request, func)*
        
        %% Internal Loop Methods
        #_recv_loop()
        #_save_loop()
        #_process_load_request(req_id)*
        #_get_next_save_task() : dict*
        #_process_save_task(task)*
    }

    %% --- Chunk Manager (Core Logic) ---
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

    %% --- Future Extension ---
    class OmniKVTransferManager {
        <<future extension>>
        <<vllm_omni.distributed.omni_connectors.kv_transfer_manager>>
        
        %% Currently independent, can extend BaseOmniChunkManager in future
        +handle_finished_requests_kv_transfer()
        +receive_kv_cache_for_request()
        +apply_kv_cache_to_request()
    }

    %% --- Connectors (Transport Layer) ---
    class OmniConnectorBase {
        <<abstract>>
        <<vllm_omni.distributed.omni_connectors.connectors.base>>
        +put(from_stage, to_stage, key, data)*
        +get(from_stage, to_stage, key)*
    }
    
    class SharedMemoryConnector {
        <<vllm_omni.distributed.omni_connectors.connectors.shm_connector>>
        +put(from_stage, to_stage, key, data)
        +get(from_stage, to_stage, key)
    }

    %% --- Relationships ---
    OmniARScheduler --> OmniChunkManager : 组合 & 委托
    OmniGenerationScheduler --> OmniChunkManager : 组合 & 委托
    OmniChunkManager --|> BaseOmniChunkManager : 继承
    OmniKVTransferManager -.-> BaseOmniChunkManager : 未来可继承
    BaseOmniChunkManager --> OmniConnectorBase : 使用基础传输
    SharedMemoryConnector --|> OmniConnectorBase : 实现
```

## Key Changes

### 1. New Base Class: `BaseOmniChunkManager`

**Location**: `vllm_omni/distributed/omni_connectors/base_chunk_manager.py`

**Purpose**: Provides common infrastructure for asynchronous chunk/cache operations

**Key Features**:
- Abstract base class for chunk management
- Request queue management (pending/finished for both load and save)
- Background thread management (_recv_loop, _save_loop)
- Abstract methods for subclass implementation
- Helper methods for state transitions

**Benefits**:
- Code reuse across different chunk/cache managers
- Clear separation of infrastructure vs. implementation
- Easier to extend for new use cases
- Better testability

### 2. Refactored: `OmniChunkManager`

**Location**: `vllm_omni/distributed/omni_connectors/adapter.py`

**Changes**:
- Now extends `BaseOmniChunkManager`
- Implements abstract methods:
  - `request_chunk(request)`: Request chunk loading
  - `submit_chunk(output, request, func)`: Submit chunk for saving
  - `_process_load_request(req_id)`: Process individual load request
  - `_get_next_save_task()`: Get next save task from queue
  - `_process_save_task(task)`: Process individual save task

**Backward Compatibility**:
- Legacy methods still available:
  - `get_chunk()` → delegates to `request_chunk()`
  - `put_chunk()` → delegates to `submit_chunk()`
  - `get_finished()` → delegates to `get_finished_load_requests()`

### 3. Scheduler Integration

**No Changes Required**: Schedulers continue to work with legacy methods

**Recommended Migration**:
```python
# Old (still works)
self.chunk_manager.get_chunk(request)
finished = self.chunk_manager.get_finished()
self.chunk_manager.put_chunk(output, request, func)

# New (recommended)
self.chunk_manager.request_chunk(request)
finished = self.chunk_manager.get_finished_load_requests()
self.chunk_manager.submit_chunk(output, request, func)
```

## Architecture Benefits

### 1. Separation of Concerns

- **Schedulers**: Focus on scheduling and request lifecycle
- **BaseOmniChunkManager**: Provides async I/O infrastructure
- **OmniChunkManager**: Implements chunk-specific logic
- **Connectors**: Handle transport layer

### 2. Extensibility

The base class design makes it easy to:
- Create new chunk managers for different use cases
- Unify OmniKVTransferManager with the base class
- Add instrumentation/monitoring
- Implement custom processing pipelines

### 3. Maintainability

- Clear abstraction boundaries
- Well-defined interfaces
- Easy to test individual components
- Self-documenting code structure

## Future Work

### 1. Unify OmniKVTransferManager

The existing `OmniKVTransferManager` can be refactored to extend `BaseOmniChunkManager`:

```python
class OmniKVTransferManager(BaseOmniChunkManager):
    """KV cache transfer using base chunk manager infrastructure."""
    
    def request_chunk(self, request):
        # Request KV cache for this request
        pass
    
    def submit_chunk(self, output, request, func):
        # Submit KV cache for transfer
        pass
```

### 2. Configurable Processing Pipeline

```python
class ConfigurableChunkManager(BaseOmniChunkManager):
    def __init__(self, connector, pipeline):
        super().__init__(connector)
        self.pipeline = pipeline
    
    def _process_load_request(self, req_id):
        data = self.connector.get(...)
        for processor in self.pipeline:
            data = processor.transform(data)
        # Apply to request
```

### 3. Metrics and Monitoring

```python
class InstrumentedChunkManager(OmniChunkManager):
    def _process_load_request(self, req_id):
        with self.metrics.timer("load_request"):
            super()._process_load_request(req_id)
```

## Migration Path

1. **Phase 1 (Current)**: 
   - Base class created
   - OmniChunkManager refactored
   - Backward compatibility maintained

2. **Phase 2 (Future)**:
   - Update schedulers to use new interface
   - Deprecate legacy methods
   - Add deprecation warnings

3. **Phase 3 (Future)**:
   - Refactor OmniKVTransferManager to extend base class
   - Unify chunk and KV cache transfer logic
   - Remove deprecated methods

## Testing Recommendations

1. **Unit Tests**:
   - Test BaseOmniChunkManager with mock connector
   - Test OmniChunkManager methods independently
   - Test thread safety and concurrency

2. **Integration Tests**:
   - Test scheduler + chunk manager + connector flow
   - Test with real SharedMemoryConnector
   - Test error handling and retries

3. **End-to-End Tests**:
   - Test full multi-stage pipeline
   - Test with different model configurations
   - Test performance and latency

## Documentation

- **Architecture**: This document
- **API Reference**: See docstrings in source files
- **Usage Examples**: See `chunk_manager_architecture.md`
- **Migration Guide**: See `chunk_manager_architecture.md`
