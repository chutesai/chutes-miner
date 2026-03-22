SHELL := /bin/bash -e -o pipefail
PROJECT ?= chutes-miner
BRANCH_NAME ?= $(shell git rev-parse --abbrev-ref HEAD | tr '/' '-')
BUILD_NUMBER ?= 0
IMAGE ?= ${PROJECT}:${BRANCH_NAME}-${BUILD_NUMBER}
COMPOSE_FILE=docker-compose.yaml
COMPOSE_BASE_FILE=docker-compose.base.yaml
DC=docker compose
SERVICE := chutes-miner
POETRY ?= "poetry"
# VERSION := $(shell head VERSION | grep -Eo "\d+.\d+.\d+")

# Monorepo configuration
SRC_DIR := src
DOCKER_DIR := docker

# Find all packages in src and docker directories
SRC_PACKAGES := $(shell find ${SRC_DIR} -maxdepth 1 -type d ! -path ${SRC_DIR} 2>/dev/null | sort)
DOCKER_PACKAGES := $(shell find ${DOCKER_DIR} -maxdepth 1 -type d ! -path ${DOCKER_DIR} 2>/dev/null | sort)

# Get package names from both directories
SRC_PACKAGE_NAMES := $(notdir $(SRC_PACKAGES))
DOCKER_PACKAGE_NAMES := $(notdir $(DOCKER_PACKAGES))

# Combine and get unique package names
ALL_PACKAGE_NAMES := $(sort $(SRC_PACKAGE_NAMES) $(DOCKER_PACKAGE_NAMES))

# Chart names (for sign-charts targeting)
ALL_CHART_NAMES := $(notdir $(shell find charts -maxdepth 1 -type d ! -path charts 2>/dev/null | sort))

# Target specific project or all projects
ifeq ($(words $(MAKECMDGOALS)),2)
	TARGET := $(word 2,$(MAKECMDGOALS))
endif

ifdef TARGET
	ifneq ($(filter $(TARGET),$(ALL_PACKAGE_NAMES)),)
		TARGET_NAMES := $(TARGET)
	endif
	ifneq ($(filter $(TARGET),$(ALL_CHART_NAMES)),)
		CHART_TARGET_NAMES := $(TARGET)
	endif
	# Target must be either a package or a chart
	ifeq ($(filter $(TARGET),$(ALL_PACKAGE_NAMES) $(ALL_CHART_NAMES)),)
		$(error $(TARGET) not found in ${SRC_DIR}, ${DOCKER_DIR}, or charts/)
	endif
else
	TARGET_NAMES := $(ALL_PACKAGE_NAMES)
endif

# Prevent Make from trying to run package/chart names as commands (dedupe: some names exist in both)
ALL_TARGET_NAMES := $(sort $(ALL_PACKAGE_NAMES) $(ALL_CHART_NAMES))
.PHONY: $(ALL_TARGET_NAMES)
$(ALL_TARGET_NAMES):
	@:

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:

include makefiles/development.mk
include makefiles/help.mk
include makefiles/lint.mk
include makefiles/local.mk
include makefiles/test.mk
include makefiles/images.mk
include makefiles/charts.mk

.PHONY: list-packages
list-packages: ##@other List all packages in the monorepo
	@echo "Available packages:"
	@for pkg in $(ALL_PACKAGE_NAMES); do \
		has_src=""; \
		has_docker=""; \
		[ -d "${SRC_DIR}/$$pkg" ] && has_src="src"; \
		[ -d "${DOCKER_DIR}/$$pkg" ] && has_docker="docker"; \
		if [ -n "$$has_src" ] && [ -n "$$has_docker" ]; then \
			locations="$$has_src, $$has_docker"; \
		else \
			locations="$$has_src$$has_docker"; \
		fi; \
		echo "  - $$pkg ($$locations)"; \
	done
	@echo ""
	@echo "Usage: make <target> <package_name>"