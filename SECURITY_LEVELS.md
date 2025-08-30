# Security Access Levels - Quick Reference

## 🔐 Access Level Hierarchy

| Level | Name | Access | Typical Content | Notion Token |
|-------|------|--------|-----------------|--------------|
| **1** | **Public** | Anyone | Marketing materials, public website content, product brochures | `NOTION_API_KEY_PUBLIC_ACCESS` |
| **2** | **Client** | Customers | User manuals, FAQs, troubleshooting guides, warranty info | `NOTION_API_KEY_CLIENT_ACCESS` |
| **3** | **Partner** | Business Partners | Integration docs, API specifications, partner resources | `NOTION_API_KEY_PARTNER_ACCESS` |
| **4** | **Employee** | Internal Staff | Support tickets, internal procedures, technical docs | `NOTION_API_KEY_EMPLOYEE_ACCESS` |
| **5** | **Management** | Leadership | Strategic plans, financial data, confidential reports | `NOTION_API_KEY_MANAGEMENT_ACCESS` |

## 🎯 How It Works

### Access Rules
- **Higher levels can see everything below**: Management (5) can see all content
- **Lower levels are restricted**: Public (1) can only see public content
- **Each document is tagged** with its required access level

### Example Access Scenarios

#### Support Ticket (NC2068)
- **Tagged as**: Level 4 (Employee)
- ✅ **Can access**: Employee, Management
- ❌ **Cannot access**: Public, Client, Partner

#### Product Manual
- **Tagged as**: Level 2 (Client)
- ✅ **Can access**: Client, Partner, Employee, Management
- ❌ **Cannot access**: Public

#### Marketing Brochure
- **Tagged as**: Level 1 (Public)
- ✅ **Can access**: Everyone

## 🔍 Search Filtering

When searching, the system automatically filters results based on user's access level:

```python
# Public user search
search(query="water dispenser", user_level=1)
→ Returns only public documents

# Employee search
search(query="water dispenser", user_level=4)
→ Returns public + client + partner + employee documents

# Management search
search(query="water dispenser", user_level=5)
→ Returns all documents
```

## 📝 Document Ingestion

When ingesting documents, the security level is automatically set based on which Notion token is used:

```python
# Using employee token
pipeline = NotionIngestionPipeline(
    security_level="employee"  # Sets access_level=4
)

# Using client token
pipeline = NotionIngestionPipeline(
    security_level="client"    # Sets access_level=2
)
```

## 🏢 Smart Water Dispenser Context

For your smart water dispenser company:

### Level 1 - Public
- Product catalog
- Marketing website content
- Press releases
- Public specifications

### Level 2 - Client
- User manuals
- Troubleshooting guides
- Warranty information
- FAQ documents

### Level 3 - Partner
- API documentation
- Integration guides
- B2B specifications
- Partner portal content

### Level 4 - Employee
- Internal support tickets (like NC2068)
- Technical documentation
- Service procedures
- Engineering specs
- Department communications

### Level 5 - Management
- Strategic plans
- Financial reports
- Board presentations
- Sensitive HR documents
- Acquisition plans

## 🚀 Implementation Status

✅ **Implemented**:
- Multi-token configuration
- Security level tagging during ingestion
- Access level metadata on all documents
- Level-based filtering ready for search

⏳ **Next Steps**:
- Add user authentication to determine access level
- Implement search filtering based on user level
- Add access audit logging
- Create role-based dashboard views