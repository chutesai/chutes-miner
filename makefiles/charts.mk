HELM_REPO_URL ?= https://chutesai.github.io/chutes-miner

.PHONY: sign-charts
sign-charts: ##@charts Sign helm charts with GPG and publish to gh-pages (set HELM_SIGN_PASSPHRASE once to avoid per-sign prompts)
sign-charts:
	@if [ -z "$$HELM_SIGN_KEY" ]; then \
		echo "Error: HELM_SIGN_KEY environment variable is not set"; \
		echo "Please set HELM_SIGN_KEY to the GPG key name or email (e.g., your.email@example.com)"; \
		exit 1; \
	fi; \
	if [ -z "$$HELM_SIGN_PASSPHRASE" ]; then \
		echo "Enter GPG key passphrase (used for all charts in this run):"; \
		read -s HELM_SIGN_PASSPHRASE; \
		export HELM_SIGN_PASSPHRASE; \
		echo ""; \
	fi; \
	export HELM_SIGN_PASSPHRASE; \
	passphrase_file=$$(mktemp); \
	trap "rm -f $$passphrase_file" EXIT; \
	echo "$$HELM_SIGN_PASSPHRASE" > $$passphrase_file; \
	chmod 600 $$passphrase_file; \
	mkdir -p build/charts; \
	keyring_args=""; \
	if [ -n "$$HELM_SIGN_KEYRING" ]; then \
		keyring_args="--keyring $$HELM_SIGN_KEYRING"; \
	elif [ -f "$$HOME/.gnupg/pubring.gpg" ]; then \
		keyring_args="--keyring $$HOME/.gnupg/pubring.gpg"; \
	elif [ -f "$$HOME/.gnupg/pubring.kbx" ]; then \
		echo "Error: Helm does not support GnuPG Keybox format (pubring.kbx)."; \
		echo "Export your signing key and set HELM_SIGN_KEYRING:"; \
		echo "  gpg --export-secret-keys \"$$HELM_SIGN_KEY\" > ~/.chutes/helm-signing-key.gpg"; \
		echo "  export HELM_SIGN_KEYRING=~/.chutes/helm-signing-key.gpg"; \
		exit 1; \
	fi; \
	all_charts=$$(find charts -maxdepth 1 -type d ! -path charts | sort); \
	if [ -n "$(CHART_TARGET_NAMES)" ]; then \
		charts=""; \
		for name in $(CHART_TARGET_NAMES); do \
			if [ -d "charts/$$name" ]; then charts="$$charts charts/$$name"; fi; \
		done; \
		charts=$$(echo $$charts | xargs); \
	else \
		charts="$$all_charts"; \
	fi; \
	echo "Signing charts: $$(echo $$charts | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"; \
	for chart_dir in $$charts; do \
		chart_name=$$(basename $$chart_dir); \
		echo "--------------------------------------------------------"; \
		echo "Signing $$chart_name"; \
		echo "--------------------------------------------------------"; \
		helm dependency build "$$chart_dir" || exit 1; \
		helm package --sign \
			--key "$$HELM_SIGN_KEY" \
			--passphrase-file "$$passphrase_file" \
			$$keyring_args \
			-d build/charts \
			"$$chart_dir" || exit 1; \
	done; \
	echo ""; \
	echo "Publishing to gh-pages..."; \
	worktree_dir="$$(pwd)/.helm-repo-worktree"; \
	trap "rm -f $$passphrase_file; git worktree remove -f '$$worktree_dir' 2>/dev/null || rm -rf '$$worktree_dir'" EXIT; \
	repo_url="$(HELM_REPO_URL)"; \
	git fetch origin gh-pages:gh-pages 2>/dev/null || true; \
	rm -rf "$$worktree_dir"; \
	git worktree add "$$worktree_dir" gh-pages 2>/dev/null || git worktree add -b gh-pages "$$worktree_dir" origin/gh-pages 2>/dev/null || { \
		echo "Error: Could not create worktree for gh-pages. Ensure origin/gh-pages exists."; \
		exit 1; \
	}; \
	cp build/charts/*.tgz build/charts/*.prov "$$worktree_dir/" 2>/dev/null || cp build/charts/*.tgz "$$worktree_dir/"; \
	if [ -f "$$worktree_dir/index.yaml" ]; then \
		helm repo index "$$worktree_dir" --merge "$$worktree_dir/index.yaml" --url "$$repo_url"; \
	else \
		helm repo index "$$worktree_dir" --url "$$repo_url"; \
	fi; \
	cd "$$worktree_dir" && \
	git add -A && \
	git status && \
	if git diff --cached --quiet 2>/dev/null; then \
		echo "No changes to publish."; \
		exit 0; \
	fi && \
	git config user.email "$${GIT_EMAIL:-helm-publish@chutes.ai}" && \
	git config user.name "$${GIT_NAME:-Helm Chart Publisher}" && \
	git commit -m "Publish signed helm charts" && \
	echo "" && \
	echo "Push to origin gh-pages? [y/N]"; \
	read -r confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		git push origin gh-pages; \
		echo "Pushed."; \
	else \
		echo "Skipped push. Signed charts are in build/charts/"; \
	fi
