set(SIBR_OURS_HEADERS
	OursView.hpp
	Config.hpp
)

set(SIBR_OURS_SOURCES
	OursView.cpp
	OursPbr.cu
	OursPbr.cuh
)

add_library(sibr_ours SHARED ${SIBR_OURS_HEADERS} ${SIBR_OURS_SOURCES})
target_compile_definitions(sibr_ours PRIVATE SIBR_EXP_OURS_EXPORTS)
target_include_directories(sibr_ours PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sibr_ours
	PUBLIC
		sibr_gaussian
		sibr_view
		sibr_assets
		sibr_renderer
		sibr_basic
		CUDA::cudart
)
set_target_properties(sibr_ours PROPERTIES DEBUG_POSTFIX "${SIBR_DEBUG_POSTFIX}" CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
