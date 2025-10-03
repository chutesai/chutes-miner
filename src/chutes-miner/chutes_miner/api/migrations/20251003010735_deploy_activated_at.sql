-- migrate:up
ALTER TABLE deployments ADD COLUMN IF NOT EXISTS activated_at TIMESTAMPTZ;
UPDATE deployments SET activated_at = NOW() WHERE active IS TRUE;

-- migrate:down
ALTER TABLE deployments DROP COLUMN IF EXISTS activated_at;
