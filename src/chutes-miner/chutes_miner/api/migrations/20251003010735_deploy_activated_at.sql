-- migrate:up
ALTER TABLE deployments ADD COLUMN IF NOT EXISTS activated_at TIMESTAMPTZ;

-- migrate:down
ALTER TABLE deployments DROP COLUMN IF EXISTS activated_at;