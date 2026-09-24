set(SIBR_OURS_APP_RESOURCES
	resources/sibr_ours_app.ini
)

add_executable(SIBR_ours_app
	main.cpp
	${SIBR_OURS_APP_RESOURCES}
)

target_link_libraries(SIBR_ours_app PRIVATE
	sibr_ours
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
)

set_target_properties(SIBR_ours_app PROPERTIES DEBUG_POSTFIX "${SIBR_DEBUG_POSTFIX}")

install(TARGETS SIBR_ours_app RUNTIME DESTINATION bin)
install(FILES ${SIBR_OURS_APP_RESOURCES} DESTINATION bin)
