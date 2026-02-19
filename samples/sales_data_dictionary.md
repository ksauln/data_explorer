# Synthetic Sales Data Dictionary (3-File Join Pack)

## Overview
This synthetic pack contains one fact table (10,000 rows) and two dimension tables for join testing.

## Files
- `sales_fact_transactions_10k.csv` (main fact file)
- `sales_dim_customers.csv` (customer dimension)
- `sales_dim_products.csv` (product dimension)

## File Details

### 1) `sales_fact_transactions_10k.csv`
- Grain: one row per sale transaction (`sale_id`).
- Row count: 10,000
- Key fields:
  - `sale_id`: unique transaction id
  - `customer_id`: foreign key to customer dimension
  - `product_id`: foreign key to product dimension
- Core metrics:
  - `quantity`, `unit_price`, `discount_pct`, `gross_sales`, `net_sales`, `cost`, `profit`

### 2) `sales_dim_customers.csv`
- Grain: one row per customer (`customer_id`).
- Row count: 2,500
- Attributes:
  - `customer_name`, `segment`, `industry`, `state`, `signup_date`, `status`, `account_manager`

### 3) `sales_dim_products.csv`
- Grain: one row per product (`product_id`).
- Row count: 400
- Attributes:
  - `product_name`, `category`, `sub_category`, `list_price`, `standard_cost`, `is_active`, `launch_date`

## Join Keys
- `sales_fact_transactions_10k.customer_id = sales_dim_customers.customer_id`
- `sales_fact_transactions_10k.product_id = sales_dim_products.product_id`

## Join Testing Notes
- Fact contains intentional orphan keys for left-join QA:
  - `customer_id = C99999`
  - `product_id = P9999`

## Example Join Queries

### Revenue by Customer Segment
```sql
SELECT
  c.segment,
  ROUND(SUM(f.net_sales), 2) AS total_net_sales
FROM sales_fact_transactions_10k f
LEFT JOIN sales_dim_customers c
  ON f.customer_id = c.customer_id
GROUP BY 1
ORDER BY total_net_sales DESC;
```

### Revenue by Product Category
```sql
SELECT
  p.category,
  ROUND(SUM(f.net_sales), 2) AS total_net_sales
FROM sales_fact_transactions_10k f
LEFT JOIN sales_dim_products p
  ON f.product_id = p.product_id
GROUP BY 1
ORDER BY total_net_sales DESC;
```

### Orphan Key Check
```sql
SELECT
  SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) AS orphan_customers,
  SUM(CASE WHEN p.product_id IS NULL THEN 1 ELSE 0 END) AS orphan_products
FROM sales_fact_transactions_10k f
LEFT JOIN sales_dim_customers c
  ON f.customer_id = c.customer_id
LEFT JOIN sales_dim_products p
  ON f.product_id = p.product_id;
```
