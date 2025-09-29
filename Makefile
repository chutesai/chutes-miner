SHELL := /bin/bash -e -o pipefail
PROJECT ?= chutes-miner
BRANCH_NAME ?= local
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

# Target specific project or all projects
ifeq ($(words $(MAKECMDGOALS)),2)
	TARGET := $(word 2,$(MAKECMDGOALS))
endif

ifdef TARGET
	# Check if target exists in either src or docker
	ifneq ($(filter $(TARGET),$(ALL_PACKAGE_NAMES)),)
		TARGET_NAMES := ${TARGET}
	else
		$(error Project ${TARGET} not found in ${SRC_DIR} or ${DOCKER_DIR})
	endif
else
	TARGET_NAMES := $(ALL_PACKAGE_NAMES)
endif

# Prevent Make from trying to run package names as commands
.PHONY: $(ALL_PACKAGE_NAMES)
$(ALL_PACKAGE_NAMES):
	@:

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:

include makefiles/development.mk
include makefiles/help.mk
include makefiles/lint.mk
include makefiles/local.mk
include makefiles/test.mk
include makefiles/images.mk

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