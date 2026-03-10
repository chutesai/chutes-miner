-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS node_selector JSONB;

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS node_selector;
