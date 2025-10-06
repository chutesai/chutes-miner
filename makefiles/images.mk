.PHONY: tag
tag: ##@images Tag docker images from the build step for Parachutes repo
tag:
	@images=$$(find docker -maxdepth 1 -type d ! -path docker | sort); \
	image_names=$$(echo $$images | xargs -n1 basename | tr '\n' ' '); \
	filtered_images=""; \
	filtered_names=""; \
	for image_dir in $$images; do \
		pkg_name=$$(basename $$image_dir); \
		if echo "$(TARGET_NAMES)" | grep -w -q "$$pkg_name"; then \
			filtered_images="$$filtered_images $$image_dir"; \
			filtered_names="$$filtered_names $$pkg_name"; \
		fi; \
	done; \
	echo "Tagging images for:$$filtered_names"; \
	for image_dir in $$filtered_images; do \
		pkg_name=$$(basename $$image_dir); \
		if [ -f "src/$$pkg_name/VERSION" ]; then \
			pkg_version=$$(head "src/$$pkg_name/VERSION"); \
		elif [ -f "$$image_dir/VERSION" ]; then \
			pkg_version=$$(head "$$image_dir/VERSION"); \
		fi; \
		echo "--------------------------------------------------------"; \
		echo "Tagging $$pkg_name (version: $$pkg_version)"; \
		echo "--------------------------------------------------------"; \
		if [ -d "docker/$$pkg_name" ]; then \
			if [ -f "docker/$$pkg_name/Dockerfile" ]; then \
				if [ -f "docker/$$pkg_name/image.conf" ]; then \
					dockerfile="docker/$$pkg_name/Dockerfile"; \
					available_targets=$$(grep -i "^FROM.*AS" $$dockerfile | sed 's/.*AS[[:space:]]*\([^[:space:]]*\).*/\1/' | tr '[:upper:]' '[:lower:]' || echo "production development"); \
					image_conf=$$(cat $$image_dir/image.conf); \
					registry=$$(echo "$$image_conf" | cut -d'/' -f1); \
					image_full_name=$$(echo "$$image_conf" | cut -d'/' -f2); \
					if [[ "$$image_full_name" == *:* ]]; then \
						image_name=$$(echo "$$image_full_name" | cut -d':' -f1); \
						image_tag="$$(echo "$$image_full_name" | cut -d':' -f2)-"; \
					else \
						image_name=$$(echo "$$image_conf" | cut -d'/' -f2); \
						image_tag=""; \
					fi; \
					if [ "${BRANCH_NAME}" != "main" ]; then \
						image_tag=$$image_tag"dev-"; \
					fi; \
					for stage_target in $$available_targets; do \
						if [[ "$$stage_target" == production* ]]; then \
							if [[ "$$stage_target" == *-* ]]; then \
								suffix=$$(echo $$stage_target | sed 's/production-//'); \
								latest_tag="$$image_tag$$suffix-latest"; \
								target_tag="$$image_tag$$suffix-$$pkg_version"; \
								src_version=$$pkg_version-$$suffix; \
							else \
								latest_tag=$$image_tag"latest"; \
								target_tag="$$image_tag$$pkg_version"; \
								src_version=$$pkg_version; \
							fi; \
							echo "docker tag $$pkg_name:$$src_version $$registry/$$image_name:$$target_tag"; \
							docker tag $$pkg_name:$$src_version $$registry/$$image_name:$$target_tag; \
							echo "docker tag $$pkg_name:$$src_version $$registry/$$image_name:$$latest_tag"; \
							docker tag $$pkg_name:$$src_version $$registry/$$image_name:$$latest_tag; \
						fi; \
					done; \
				else \
					echo "Skipping $$pkg_name: docker/$$pkg_name/image.conf not found"; \
				fi; \
			else \
				echo "Skipping $$pkg_name: docker/$$pkg_name/Dockerfile not found"; \
			fi; \
		else \
			echo "Skipping $$pkg_name: docker/$$pkg_name directory not found"; \
		fi; \
		echo ; \
	done

.PHONY: push
push: ##@images Tag docker images from the build step for Parachutes repo
push:
push:
	@images=$$(find docker -maxdepth 1 -type d ! -path docker | sort); \
	image_names=$$(echo $$images | xargs -n1 basename | tr '\n' ' '); \
	filtered_images=""; \
	filtered_names=""; \
	for image_dir in $$images; do \
		pkg_name=$$(basename $$image_dir); \
		if echo "$(TARGET_NAMES)" | grep -w -q "$$pkg_name"; then \
			filtered_images="$$filtered_images $$image_dir"; \
			filtered_names="$$filtered_names $$pkg_name"; \
		fi; \
	done; \
	echo "Pushing images for:$$filtered_names"; \
	for image_dir in $$filtered_images; do \
		pkg_name=$$(basename $$image_dir); \
		if [ -f "src/$$pkg_name/VERSION" ]; then \
			pkg_version=$$(head "src/$$pkg_name/VERSION"); \
		elif [ -f "$$image_dir/VERSION" ]; then \
			pkg_version=$$(head "$$image_dir/VERSION"); \
		fi; \
		echo "--------------------------------------------------------"; \
		echo "Pushing $$pkg_name (version: $$pkg_version)"; \
		echo "--------------------------------------------------------"; \
		if [ -d "docker/$$pkg_name" ]; then \
			if [ -f "docker/$$pkg_name/Dockerfile" ]; then \
				if [ -f "docker/$$pkg_name/image.conf" ]; then \
					dockerfile="docker/$$pkg_name/Dockerfile"; \
					available_targets=$$(grep -i "^FROM.*AS" $$dockerfile | sed 's/.*AS[[:space:]]*\([^[:space:]]*\).*/\1/' | tr '[:upper:]' '[:lower:]' || echo "production development"); \
					image_conf=$$(cat $$image_dir/image.conf); \
					registry=$$(echo "$$image_conf" | cut -d'/' -f1); \
					image_full_name=$$(echo "$$image_conf" | cut -d'/' -f2); \
					if [[ "$$image_full_name" == *:* ]]; then \
						image_name=$$(echo "$$image_full_name" | cut -d':' -f1); \
						image_tag="$$(echo "$$image_full_name" | cut -d':' -f2)-"; \
					else \
						image_name=$$(echo "$$image_conf" | cut -d'/' -f2); \
						image_tag=""; \
					fi; \
					if [ "${BRANCH_NAME}" != "main" ]; then \
						image_tag=$$image_tag"dev-"; \
					fi; \
					for stage_target in $$available_targets; do \
						if [[ "$$stage_target" == production* ]]; then \
							if [[ "$$stage_target" == *-* ]]; then \
								suffix=$$(echo $$stage_target | sed 's/production-//'); \
								latest_tag="$$image_tag$$suffix-latest"; \
								target_tag="$$image_tag$$suffix-$$pkg_version"; \
							else \
								latest_tag=$$image_tag"latest"; \
								target_tag="$$image_tag$$pkg_version"; \
							fi; \
							echo "docker push $$registry/$$image_name:$$target_tag"; \
							docker push $$registry/$$image_name:$$target_tag; \
							echo "docker push $$registry/$$image_name:$$latest_tag"; \
							docker push $$registry/$$image_name:$$latest_tag; \
						fi; \
					done; \
				else \
					echo "Skipping $$pkg_name: docker/$$pkg_name/image.conf not found"; \
				fi; \
			else \
				echo "Skipping $$pkg_name: docker/$$pkg_name/Dockerfile not found"; \
			fi; \
		else \
			echo "Skipping $$pkg_name: docker/$$pkg_name directory not found"; \
		fi; \
		echo ; \
	done

.PHONY: images
images: ##@images Build all docker images
images: args ?= --network=host --build-arg BUILDKIT_INLINE_CACHE=1
images:
	@images=$$(find docker -maxdepth 1 -type d ! -path docker | sort); \
	filtered_images=""; \
	for image_dir in $$images; do \
		pkg_name=$$(basename $$image_dir); \
		if ! echo "$(TARGET_NAMES)" | grep -q "$$pkg_name"; then \
			filtered_images="$$filtered_images $$image_dir"; \
		fi; \
	done; \
	image_names=$$(echo $$filtered_images | xargs -n1 basename | tr '\n' ' '); \
	echo "Building images for: $$image_names"; \
	for image_dir in $$filtered_images; do \
		pkg_name=$$(basename $$image_dir); \
		pkg_version=$$(if [ -f "$$image_dir/VERSION" ]; then head "$$image_dir/VERSION"; else echo "dev"; fi); \
		if [ -f "$$image_dir/Dockerfile" ]; then \
			echo "Building images for $$pkg_name (version: $$pkg_version)"; \
			DOCKER_BUILDKIT=1 docker build --progress=plain --target production \
				-f $$image_dir/Dockerfile \
				-t $$pkg_name:${BRANCH_NAME}-${BUILD_NUMBER} \
				-t $$pkg_name:$$pkg_version \
				--build-arg PROJECT_DIR=$$pkg_name \
				--build-arg PROJECT=$$pkg_name \
				${args} .; \
			DOCKER_BUILDKIT=1 docker build --progress=plain --target development \
				-f $$image_dir/Dockerfile \
				-t $$pkg_name\_development:${BRANCH_NAME}-${BUILD_NUMBER} \
				-t $$pkg_name\_development:$$pkg_version \
				--build-arg PROJECT_DIR=$$pkg_name \
				--build-arg PROJECT=$$pkg_name \
				--cache-from $$pkg_name:${BRANCH_NAME}-${BUILD_NUMBER} \
				${args} .; \
		else \
			echo "Skipping $$pkg_name: $$image_dir/Dockerfile not found"; \
		fi; \
	done