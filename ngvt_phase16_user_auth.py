#!/usr/bin/env python3
"""
Phase 16: User Management & Authentication System
Confucius SDK v2.2

Advanced user management with:
- User accounts and profiles
- Role-based access control (RBAC)
- API token generation and management
- Permission enforcement
- Session management
- Audit logging
- Multi-factor authentication support

Author: Confucius SDK Development Team
Version: 2.2.0
"""

import hashlib
import secrets
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import re


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class UserRole(Enum):
    """User role definitions"""
    ADMIN = "admin"                    # Full system access
    DEVELOPER = "developer"            # API access, read/write
    ANALYST = "analyst"                # Read-only analytics access
    VIEWER = "viewer"                  # Read-only access
    SERVICE_ACCOUNT = "service_account" # Automated integrations


class PermissionAction(Enum):
    """Permission actions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_TOKENS = "manage_tokens"
    VIEW_ANALYTICS = "view_analytics"
    CONFIGURE_SYSTEM = "configure_system"
    AUDIT_LOG = "audit_log"


class TokenType(Enum):
    """API token types"""
    API_KEY = "api_key"               # Long-lived API keys
    SESSION_TOKEN = "session_token"    # Short-lived session tokens
    REFRESH_TOKEN = "refresh_token"    # Used to refresh session tokens
    SERVICE_TOKEN = "service_token"    # Service-to-service tokens


class AuditAction(Enum):
    """Audit log actions"""
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    PERMISSION_DENIED = "permission_denied"
    SYSTEM_CONFIGURED = "system_configured"


# ============================================================================
# PERMISSION AND ROLE SYSTEM
# ============================================================================

@dataclass
class Permission:
    """Single permission definition"""
    action: PermissionAction
    resource: str  # e.g., "api", "analytics", "users"
    description: str = ""


class PermissionSet:
    """Manages a set of permissions"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.permissions: List[Permission] = []
    
    def add_permission(self, action: PermissionAction, resource: str, description: str = ""):
        """Add a permission"""
        perm = Permission(action, resource, description)
        if perm not in self.permissions:
            self.permissions.append(perm)
    
    def has_permission(self, action: PermissionAction, resource: str) -> bool:
        """Check if permission exists"""
        return any(
            p.action == action and p.resource == resource
            for p in self.permissions
        )
    
    def get_all_permissions(self) -> List[Tuple[str, str]]:
        """Get all permissions as (action, resource) tuples"""
        return [(p.action.value, p.resource) for p in self.permissions]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [
                {"action": p.action.value, "resource": p.resource, "description": p.description}
                for p in self.permissions
            ]
        }


class RoleManager:
    """Manages roles and their permissions"""
    
    def __init__(self):
        self.roles: Dict[str, PermissionSet] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Create default roles"""
        # Admin: Full access
        admin = PermissionSet("Admin", "Full system access")
        admin.add_permission(PermissionAction.READ, "*")
        admin.add_permission(PermissionAction.WRITE, "*")
        admin.add_permission(PermissionAction.DELETE, "*")
        admin.add_permission(PermissionAction.MANAGE_USERS, "*")
        admin.add_permission(PermissionAction.MANAGE_ROLES, "*")
        admin.add_permission(PermissionAction.MANAGE_TOKENS, "*")
        admin.add_permission(PermissionAction.VIEW_ANALYTICS, "*")
        admin.add_permission(PermissionAction.CONFIGURE_SYSTEM, "*")
        admin.add_permission(PermissionAction.AUDIT_LOG, "*")
        self.roles[UserRole.ADMIN.value] = admin
        
        # Developer: API access
        developer = PermissionSet("Developer", "API access and integrations")
        developer.add_permission(PermissionAction.READ, "api")
        developer.add_permission(PermissionAction.WRITE, "api")
        developer.add_permission(PermissionAction.READ, "analytics")
        developer.add_permission(PermissionAction.MANAGE_TOKENS, "own")
        self.roles[UserRole.DEVELOPER.value] = developer
        
        # Analyst: Read-only analytics
        analyst = PermissionSet("Analyst", "Analytics read-only access")
        analyst.add_permission(PermissionAction.READ, "analytics")
        analyst.add_permission(PermissionAction.VIEW_ANALYTICS, "all")
        self.roles[UserRole.ANALYST.value] = analyst
        
        # Viewer: Read-only
        viewer = PermissionSet("Viewer", "Read-only access")
        viewer.add_permission(PermissionAction.READ, "analytics")
        self.roles[UserRole.VIEWER.value] = viewer
        
        # Service Account: Limited API access
        service = PermissionSet("Service Account", "Service-to-service integrations")
        service.add_permission(PermissionAction.READ, "api")
        service.add_permission(PermissionAction.WRITE, "api")
        self.roles[UserRole.SERVICE_ACCOUNT.value] = service
    
    def get_role_permissions(self, role: str) -> Optional[PermissionSet]:
        """Get permissions for a role"""
        return self.roles.get(role)
    
    def can_perform_action(self, role: str, action: PermissionAction, resource: str) -> bool:
        """Check if role can perform action on resource"""
        role_perms = self.roles.get(role)
        if not role_perms:
            return False
        
        # Check exact match
        if role_perms.has_permission(action, resource):
            return True
        
        # Check wildcard match
        if role_perms.has_permission(action, "*"):
            return True
        
        return False
    
    def list_roles(self) -> Dict[str, Dict[str, Any]]:
        """List all roles with their permissions"""
        return {
            role: perm_set.to_dict()
            for role, perm_set in self.roles.items()
        }


# ============================================================================
# API TOKEN SYSTEM
# ============================================================================

@dataclass
class APIToken:
    """API token representation"""
    token_id: str
    token_hash: str  # SHA-256 hash of actual token
    token_prefix: str  # First 8 chars for display
    user_id: str
    token_type: TokenType
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    name: str = ""
    scopes: List[str] = field(default_factory=list)
    ip_restrictions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "token_id": self.token_id,
            "token_prefix": self.token_prefix,
            "user_id": self.user_id,
            "token_type": self.token_type.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
            "name": self.name,
            "scopes": self.scopes,
            "ip_restrictions": self.ip_restrictions
        }


class TokenManager:
    """Manages API tokens"""
    
    def __init__(self):
        self.tokens: Dict[str, APIToken] = {}  # token_hash -> APIToken
        self.user_tokens: Dict[str, List[str]] = {}  # user_id -> [token_hashes]
    
    def _generate_token(self, prefix: str = "cfapi") -> Tuple[str, str]:
        """Generate a new token and its hash"""
        # Format: prefix_randomstring (64 bytes = 88 chars in base64-like format)
        random_part = secrets.token_urlsafe(48)
        token = f"{prefix}_{random_part}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return token, token_hash
    
    def create_token(
        self,
        user_id: str,
        token_type: TokenType = TokenType.API_KEY,
        name: str = "",
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        ip_restrictions: Optional[List[str]] = None
    ) -> Tuple[str, APIToken]:
        """Create a new API token"""
        token, token_hash = self._generate_token()
        token_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        elif token_type == TokenType.SESSION_TOKEN:
            expires_at = datetime.utcnow() + timedelta(hours=1)
        elif token_type == TokenType.REFRESH_TOKEN:
            expires_at = datetime.utcnow() + timedelta(days=7)
        else:  # API_KEY
            expires_at = datetime.utcnow() + timedelta(days=365)
        
        api_token = APIToken(
            token_id=token_id,
            token_hash=token_hash,
            token_prefix=token[:8],
            user_id=user_id,
            token_type=token_type,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_active=True,
            name=name,
            scopes=scopes or [],
            ip_restrictions=ip_restrictions or []
        )
        
        self.tokens[token_hash] = api_token
        if user_id not in self.user_tokens:
            self.user_tokens[user_id] = []
        self.user_tokens[user_id].append(token_hash)
        
        return token, api_token
    
    def validate_token(self, token: str, ip_address: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a token
        Returns: (is_valid, user_id, error_message)
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        if token_hash not in self.tokens:
            return False, None, "Token not found"
        
        api_token = self.tokens[token_hash]
        
        if not api_token.is_active:
            return False, None, "Token is inactive"
        
        if api_token.expires_at and datetime.utcnow() > api_token.expires_at:
            return False, None, "Token expired"
        
        if api_token.ip_restrictions and ip_address:
            if ip_address not in api_token.ip_restrictions:
                return False, None, f"IP {ip_address} not in whitelist"
        
        # Update last used
        api_token.last_used_at = datetime.utcnow()
        
        return True, api_token.user_id, None
    
    def revoke_token(self, token_hash: str) -> bool:
        """Revoke a token"""
        if token_hash in self.tokens:
            self.tokens[token_hash].is_active = False
            return True
        return False
    
    def revoke_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user"""
        count = 0
        if user_id in self.user_tokens:
            for token_hash in self.user_tokens[user_id]:
                if token_hash in self.tokens:
                    self.tokens[token_hash].is_active = False
                    count += 1
        return count
    
    def get_user_tokens(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all tokens for a user"""
        if user_id not in self.user_tokens:
            return []
        
        tokens = []
        for token_hash in self.user_tokens[user_id]:
            if token_hash in self.tokens:
                tokens.append(self.tokens[token_hash].to_dict())
        return tokens
    
    def get_token_info(self, token_hash: str) -> Optional[Dict[str, Any]]:
        """Get token information"""
        if token_hash in self.tokens:
            return self.tokens[token_hash].to_dict()
        return None


# ============================================================================
# USER AND SESSION MANAGEMENT
# ============================================================================

@dataclass
class UserProfile:
    """User profile information"""
    user_id: str
    username: str
    email: str
    full_name: str = ""
    organization: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    is_active: bool = True
    mfa_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "organization": self.organization,
            "created_at": self.created_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled
        }


@dataclass
class UserSession:
    """Active user session"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active
        }


@dataclass
class AuditLogEntry:
    """Audit log entry"""
    entry_id: str
    timestamp: datetime
    user_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str = ""
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "success": self.success
        }


class AuditLogger:
    """Manages audit logging"""
    
    def __init__(self, max_entries: int = 10000):
        self.entries: List[AuditLogEntry] = []
        self.max_entries = max_entries
    
    def log_action(
        self,
        user_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
        success: bool = True
    ) -> str:
        """Log an action"""
        entry_id = str(uuid.uuid4())
        entry = AuditLogEntry(
            entry_id=entry_id,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            success=success
        )
        
        self.entries.append(entry)
        
        # Keep only recent entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        return entry_id
    
    def get_user_activity(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get activity for a user"""
        user_entries = [e for e in self.entries if e.user_id == user_id]
        return [e.to_dict() for e in user_entries[-limit:]]
    
    def get_resource_activity(self, resource_type: str, resource_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get activity for a resource"""
        resource_entries = [
            e for e in self.entries
            if e.resource_type == resource_type and e.resource_id == resource_id
        ]
        return [e.to_dict() for e in resource_entries[-limit:]]
    
    def get_failed_login_attempts(self, user_id: str, hours: int = 24) -> int:
        """Get failed login attempts for a user in the past N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        count = 0
        for entry in self.entries:
            if (entry.user_id == user_id and
                entry.action == AuditAction.LOGIN_FAILED and
                entry.timestamp > cutoff):
                count += 1
        return count
    
    def get_all_entries(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all audit entries"""
        return [e.to_dict() for e in self.entries[-limit:]]


class UserManager:
    """Manages users, sessions, and authentication"""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}  # user_id -> UserProfile
        self.username_index: Dict[str, str] = {}  # username -> user_id
        self.email_index: Dict[str, str] = {}     # email -> user_id
        self.password_hashes: Dict[str, str] = {} # user_id -> password_hash
        self.user_roles: Dict[str, List[str]] = {}  # user_id -> [roles]
        self.sessions: Dict[str, UserSession] = {}  # session_id -> UserSession
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        self.role_manager = RoleManager()
        self.token_manager = TokenManager()
        self.audit_logger = AuditLogger()
    
    def _hash_password(self, password: str) -> str:
        """Hash a password with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, pwd_hash = password_hash.split('$')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return new_hash.hex() == pwd_hash
        except:
            return False
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = "",
        organization: str = "",
        initial_roles: Optional[List[str]] = None
    ) -> Tuple[bool, str, Optional[UserProfile]]:
        """Create a new user"""
        # Validate input
        if not re.match(r'^[a-zA-Z0-9_-]{3,32}$', username):
            return False, "Invalid username format", None
        
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return False, "Invalid email format", None
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters", None
        
        # Check uniqueness
        if username in self.username_index:
            return False, "Username already exists", None
        
        if email in self.email_index:
            return False, "Email already exists", None
        
        # Create user
        user_id = str(uuid.uuid4())
        user = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            organization=organization
        )
        
        self.users[user_id] = user
        self.username_index[username] = user_id
        self.email_index[email] = user_id
        self.password_hashes[user_id] = self._hash_password(password)
        self.user_roles[user_id] = initial_roles or [UserRole.VIEWER.value]
        self.user_sessions[user_id] = []
        
        # Audit
        self.audit_logger.log_action(
            user_id=user_id,
            action=AuditAction.USER_CREATED,
            resource_type="user",
            resource_id=user_id,
            details={"username": username, "email": email, "roles": self.user_roles[user_id]},
            success=True
        )
        
        return True, "User created successfully", user
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str = ""
    ) -> Tuple[bool, Optional[str], Optional[UserProfile]]:
        """Authenticate a user"""
        # Find user
        if username not in self.username_index:
            self.audit_logger.log_action(
                user_id="unknown",
                action=AuditAction.LOGIN_FAILED,
                resource_type="user",
                resource_id="unknown",
                details={"reason": "user_not_found", "username": username},
                ip_address=ip_address,
                success=False
            )
            return False, "Invalid credentials", None
        
        user_id = self.username_index[username]
        user = self.users[user_id]
        
        # Check if user is active
        if not user.is_active:
            self.audit_logger.log_action(
                user_id=user_id,
                action=AuditAction.LOGIN_FAILED,
                resource_type="user",
                resource_id=user_id,
                details={"reason": "user_inactive"},
                ip_address=ip_address,
                success=False
            )
            return False, "User account is inactive", None
        
        # Verify password
        if not self._verify_password(password, self.password_hashes[user_id]):
            self.audit_logger.log_action(
                user_id=user_id,
                action=AuditAction.LOGIN_FAILED,
                resource_type="user",
                resource_id=user_id,
                details={"reason": "invalid_password"},
                ip_address=ip_address,
                success=False
            )
            return False, "Invalid credentials", None
        
        # Check for brute force attacks
        failed_attempts = self.audit_logger.get_failed_login_attempts(user_id, hours=1)
        if failed_attempts > 5:
            self.audit_logger.log_action(
                user_id=user_id,
                action=AuditAction.LOGIN_FAILED,
                resource_type="user",
                resource_id=user_id,
                details={"reason": "too_many_failed_attempts"},
                ip_address=ip_address,
                success=False
            )
            return False, "Too many failed login attempts. Please try again later.", None
        
        # Success
        user.last_login_at = datetime.utcnow()
        self.audit_logger.log_action(
            user_id=user_id,
            action=AuditAction.LOGIN_SUCCESS,
            resource_type="user",
            resource_id=user_id,
            details={"username": username},
            ip_address=ip_address,
            success=True
        )
        
        return True, "Authentication successful", user
    
    def create_session(
        self,
        user_id: str,
        ip_address: str = "",
        user_agent: str = "",
        expires_in_hours: int = 8
    ) -> Tuple[str, UserSession]:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        return session_id, session
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate a session"""
        if session_id not in self.sessions:
            return False, "Session not found"
        
        session = self.sessions[session_id]
        
        if not session.is_active:
            return False, "Session is inactive"
        
        if session.is_expired():
            session.is_active = False
            return False, "Session expired"
        
        return True, session.user_id
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            return True
        return False
    
    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        count = 0
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id]:
                if session_id in self.sessions:
                    self.sessions[session_id].is_active = False
                    count += 1
        return count
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign a role to a user"""
        if user_id not in self.users:
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            
            self.audit_logger.log_action(
                user_id=user_id,
                action=AuditAction.ROLE_ASSIGNED,
                resource_type="user",
                resource_id=user_id,
                details={"role": role}
            )
            return True
        return False
    
    def remove_role(self, user_id: str, role: str) -> bool:
        """Remove a role from a user"""
        if user_id in self.user_roles and role in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role)
            
            self.audit_logger.log_action(
                user_id=user_id,
                action=AuditAction.ROLE_REMOVED,
                resource_type="user",
                resource_id=user_id,
                details={"role": role}
            )
            return True
        return False
    
    def get_user_permissions(self, user_id: str) -> List[Tuple[str, str]]:
        """Get all permissions for a user"""
        if user_id not in self.user_roles:
            return []
        
        permissions = set()
        for role in self.user_roles[user_id]:
            role_perms = self.role_manager.get_role_permissions(role)
            if role_perms:
                permissions.update(role_perms.get_all_permissions())
        
        return list(permissions)
    
    def can_user_perform_action(
        self,
        user_id: str,
        action: PermissionAction,
        resource: str
    ) -> bool:
        """Check if user can perform an action"""
        if user_id not in self.user_roles:
            return False
        
        for role in self.user_roles[user_id]:
            if self.role_manager.can_perform_action(role, action, resource):
                return True
        
        return False
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get user by username"""
        if username in self.username_index:
            return self.users.get(self.username_index[username])
        return None
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive user information"""
        user = self.get_user(user_id)
        if not user:
            return None
        
        return {
            "profile": user.to_dict(),
            "roles": self.user_roles.get(user_id, []),
            "permissions": self.get_user_permissions(user_id),
            "active_sessions": len([s for s in self.user_sessions.get(user_id, [])
                                   if s in self.sessions and self.sessions[s].is_active]),
            "api_tokens": len(self.token_manager.get_user_tokens(user_id))
        }
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user information"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        allowed_fields = ['full_name', 'organization', 'is_active', 'mfa_enabled']
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(user, field, value)
        
        self.audit_logger.log_action(
            user_id=user_id,
            action=AuditAction.USER_UPDATED,
            resource_type="user",
            resource_id=user_id,
            details=kwargs
        )
        
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        # Revoke all sessions and tokens
        self.revoke_all_user_sessions(user_id)
        self.token_manager.revoke_user_tokens(user_id)
        
        # Remove from indices
        if user.username in self.username_index:
            del self.username_index[user.username]
        if user.email in self.email_index:
            del self.email_index[user.email]
        
        # Remove user
        del self.users[user_id]
        if user_id in self.password_hashes:
            del self.password_hashes[user_id]
        if user_id in self.user_roles:
            del self.user_roles[user_id]
        
        self.audit_logger.log_action(
            user_id=user_id,
            action=AuditAction.USER_DELETED,
            resource_type="user",
            resource_id=user_id
        )
        
        return True
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users"""
        return [user.to_dict() for user in self.users.values()]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        active_users = len([u for u in self.users.values() if u.is_active])
        active_sessions = len([s for s in self.sessions.values() if s.is_active and not s.is_expired()])
        total_tokens = len(self.token_manager.tokens)
        active_tokens = len([t for t in self.token_manager.tokens.values() if t.is_active])
        
        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_tokens": total_tokens,
            "active_tokens": active_tokens,
            "audit_entries": len(self.audit_logger.entries),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class UserAuthOrchestrator:
    """Main orchestrator for user management and authentication"""
    
    def __init__(self):
        self.user_manager = UserManager()
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the system"""
        return {
            "status": "initialized",
            "role_manager": "ready",
            "token_manager": "ready",
            "user_manager": "ready",
            "audit_logger": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system report"""
        return {
            "system": self.user_manager.get_system_stats(),
            "roles": self.user_manager.role_manager.list_roles(),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# DEMO AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 16: USER MANAGEMENT & AUTHENTICATION SYSTEM - DEMO")
    print("=" * 80)
    
    orchestrator = UserAuthOrchestrator()
    um = orchestrator.user_manager
    
    # Initialize
    print("\n1. SYSTEM INITIALIZATION")
    print("-" * 80)
    init_result = orchestrator.initialize()
    print(f"Status: {init_result['status']}")
    print(f"Components: {', '.join([k for k in init_result.keys() if k.endswith('_manager') or k == 'audit_logger'])}")
    
    # Create users
    print("\n2. USER CREATION")
    print("-" * 80)
    users_to_create = [
        ("alice", "alice@company.com", "SecurePass123", "Alice Admin", [UserRole.ADMIN.value]),
        ("bob", "bob@company.com", "SecurePass456", "Bob Developer", [UserRole.DEVELOPER.value]),
        ("charlie", "charlie@company.com", "SecurePass789", "Charlie Analyst", [UserRole.ANALYST.value]),
        ("diana", "diana@company.com", "SecurePass101", "Diana Viewer", [UserRole.VIEWER.value]),
    ]
    
    created_users = {}
    for username, email, password, full_name, roles in users_to_create:
        success, msg, user = um.create_user(username, email, password, full_name, initial_roles=roles)
        if success:
            created_users[username] = user.user_id
            print(f"✓ Created {username} ({email}) - Roles: {roles}")
        else:
            print(f"✗ Failed to create {username}: {msg}")
    
    # Authentication
    print("\n3. AUTHENTICATION & SESSIONS")
    print("-" * 80)
    success, msg, user = um.authenticate("alice", "SecurePass123", ip_address="192.168.1.100")
    if success:
        print(f"✓ Authenticated {user.username}")
        session_id, session = um.create_session(user.user_id, ip_address="192.168.1.100", user_agent="Chrome/91.0")
        print(f"✓ Session created: {session_id[:8]}...")
        print(f"  Expires: {session.expires_at.isoformat()}")
    
    # API Token Management
    print("\n4. API TOKEN MANAGEMENT")
    print("-" * 80)
    token, token_obj = um.token_manager.create_token(
        created_users["bob"],
        token_type=TokenType.API_KEY,
        name="Production API Key",
        expires_in_days=90,
        scopes=["read", "write"]
    )
    print(f"✓ Created API token for bob")
    print(f"  Token prefix: {token_obj.token_prefix}...")
    print(f"  Type: {token_obj.token_type.value}")
    print(f"  Expires: {token_obj.expires_at.isoformat()}")
    print(f"  Scopes: {token_obj.scopes}")
    
    # Validate token
    is_valid, user_id, error = um.token_manager.validate_token(token)
    if is_valid:
        print(f"✓ Token validated for user {user_id}")
    else:
        print(f"✗ Token validation failed: {error}")
    
    # Role and Permission Management
    print("\n5. ROLE & PERMISSION MANAGEMENT")
    print("-" * 80)
    alice_id = created_users["alice"]
    permissions = um.get_user_permissions(alice_id)
    print(f"✓ Alice permissions ({len(permissions)} total):")
    for action, resource in permissions[:5]:
        print(f"  - {action}:{resource}")
    if len(permissions) > 5:
        print(f"  ... and {len(permissions) - 5} more")
    
    # Check specific permissions
    print("\n  Permission checks:")
    can_manage_users = um.can_user_perform_action(alice_id, PermissionAction.MANAGE_USERS, "*")
    print(f"  - Can manage users: {can_manage_users}")
    can_view_analytics = um.can_user_perform_action(alice_id, PermissionAction.VIEW_ANALYTICS, "*")
    print(f"  - Can view analytics: {can_view_analytics}")
    
    bob_id = created_users["bob"]
    can_manage_users = um.can_user_perform_action(bob_id, PermissionAction.MANAGE_USERS, "*")
    print(f"  - Bob can manage users: {can_manage_users}")
    can_write_api = um.can_user_perform_action(bob_id, PermissionAction.WRITE, "api")
    print(f"  - Bob can write to API: {can_write_api}")
    
    # User Information
    print("\n6. USER INFORMATION & AUDIT")
    print("-" * 80)
    user_info = um.get_user_info(alice_id)
    print(f"✓ Alice comprehensive info:")
    print(f"  - Username: {user_info['profile']['username']}")
    print(f"  - Email: {user_info['profile']['email']}")
    print(f"  - Roles: {user_info['roles']}")
    print(f"  - Active sessions: {user_info['active_sessions']}")
    print(f"  - API tokens: {user_info['api_tokens']}")
    
    # Audit log
    print("\n  Recent audit entries for Alice:")
    audit_entries = um.audit_logger.get_user_activity(alice_id, limit=5)
    for entry in audit_entries[-3:]:
        print(f"  - {entry['action']}: {entry['resource_type']} ({entry['timestamp'][:19]})")
    
    # System Statistics
    print("\n7. SYSTEM STATISTICS")
    print("-" * 80)
    stats = um.get_system_stats()
    print(f"Total users: {stats['total_users']}")
    print(f"Active users: {stats['active_users']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"Total API tokens: {stats['total_tokens']}")
    print(f"Active API tokens: {stats['active_tokens']}")
    print(f"Audit log entries: {stats['audit_entries']}")
    
    # Comprehensive report
    print("\n8. COMPREHENSIVE SYSTEM REPORT")
    print("-" * 80)
    report = orchestrator.get_system_report()
    print(f"System Status:")
    print(f"  - Users: {report['system']['total_users']} ({report['system']['active_users']} active)")
    print(f"  - Sessions: {report['system']['total_sessions']} ({report['system']['active_sessions']} active)")
    print(f"  - API Tokens: {report['system']['total_tokens']} ({report['system']['active_tokens']} active)")
    print(f"  - Audit Entries: {report['system']['audit_entries']}")
    print(f"\nRoles Available: {len(report['roles'])}")
    for role_name, role_info in report['roles'].items():
        print(f"  - {role_name}: {len(role_info['permissions'])} permissions")
    
    print("\n" + "=" * 80)
    print("PHASE 16 DEMO COMPLETE - ALL SYSTEMS OPERATIONAL")
    print("=" * 80)
