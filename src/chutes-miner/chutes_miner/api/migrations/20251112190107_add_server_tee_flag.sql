-- migrate:up
ALTER TABLE servers ADD COLUMN is_tee BOOLEAN DEFAULT false;

-- migrate:down
ALTER TABLE servers DROP COLUMN is_tee;

