-- Настройка базы данных для E-commerce RL Recommendation System
-- PostgreSQL 17 Configuration

ALTER DATABASE ecommerce_rl SET timezone TO 'Europe/Moscow';
ALTER DATABASE ecommerce_rl SET datestyle TO 'ISO, DMY';

COMMENT ON DATABASE ecommerce_rl IS 'E-commerce Reinforcement Learning Recommendation System Database';

SELECT 'PostgreSQL 17 Database ecommerce_rl initialized successfully' AS status;