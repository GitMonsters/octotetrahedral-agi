#!/usr/bin/env python3
"""
Phase 16: User Management & Authentication - Comprehensive Test Suite

Tests for all user management, authentication, and authorization components:
- User creation and management
- Password hashing and authentication
- Session management
- API token generation and validation
- Role-based access control (RBAC)
- Permission management
- Audit logging

Author: Confucius SDK Development Team
Version: 2.2.0
"""

import unittest
from datetime import datetime, timedelta
from ngvt_phase16_user_auth import (
    UserRole, PermissionAction, TokenType, AuditAction,
    UserManager, TokenManager, RoleManager, AuditLogger,
    PermissionSet, UserAuthOrchestrator
)


class TestUserCreation(unittest.TestCase):
    """Test user creation and validation"""
    
    def setUp(self):
        self.um = UserManager()
    
    def test_create_valid_user(self):
        """Test creating a valid user"""
        success, msg, user = self.um.create_user(
            "testuser", "test@example.com", "password123"
        )
        self.assertTrue(success)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "testuser")
        self.assertEqual(user.email, "test@example.com")
    
    def test_create_user_duplicate_username(self):
        """Test preventing duplicate usernames"""
        self.um.create_user("alice", "alice@example.com", "password123")
        success, msg, user = self.um.create_user(
            "alice", "alice2@example.com", "password123"
        )
        self.assertFalse(success)
        self.assertIn("already exists", msg)
    
    def test_create_user_duplicate_email(self):
        """Test preventing duplicate emails"""
        self.um.create_user("alice", "alice@example.com", "password123")
        success, msg, user = self.um.create_user(
            "alice2", "alice@example.com", "password123"
        )
        self.assertFalse(success)
        self.assertIn("already exists", msg)
    
    def test_create_user_invalid_username(self):
        """Test invalid username format"""
        success, msg, user = self.um.create_user(
            "a", "test@example.com", "password123"
        )
        self.assertFalse(success)
        self.assertIn("Invalid username", msg)
    
    def test_create_user_invalid_email(self):
        """Test invalid email format"""
        success, msg, user = self.um.create_user(
            "testuser", "invalid-email", "password123"
        )
        self.assertFalse(success)
        self.assertIn("Invalid email", msg)
    
    def test_create_user_weak_password(self):
        """Test weak password rejection"""
        success, msg, user = self.um.create_user(
            "testuser", "test@example.com", "weak"
        )
        self.assertFalse(success)
        self.assertIn("at least 8 characters", msg)
    
    def test_create_user_with_roles(self):
        """Test creating user with initial roles"""
        success, msg, user = self.um.create_user(
            "testuser", "test@example.com", "password123",
            initial_roles=[UserRole.DEVELOPER.value]
        )
        self.assertTrue(success)
        self.assertEqual(self.um.user_roles[user.user_id], [UserRole.DEVELOPER.value])


class TestAuthentication(unittest.TestCase):
    """Test authentication and password handling"""
    
    def setUp(self):
        self.um = UserManager()
        self.um.create_user("alice", "alice@example.com", "Password123")
    
    def test_authenticate_success(self):
        """Test successful authentication"""
        success, msg, user = self.um.authenticate("alice", "Password123")
        self.assertTrue(success)
        self.assertEqual(user.username, "alice")
    
    def test_authenticate_invalid_password(self):
        """Test authentication with wrong password"""
        success, msg, user = self.um.authenticate("alice", "WrongPassword")
        self.assertFalse(success)
        self.assertIsNone(user)
    
    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user"""
        success, msg, user = self.um.authenticate("nonexistent", "password")
        self.assertFalse(success)
        self.assertIsNone(user)
    
    def test_authenticate_inactive_user(self):
        """Test authentication with inactive user"""
        alice_id = self.um.username_index["alice"]
        self.um.update_user(alice_id, is_active=False)
        success, msg, user = self.um.authenticate("alice", "Password123")
        self.assertFalse(success)
        self.assertIn("inactive", msg)
    
    def test_authenticate_updates_last_login(self):
        """Test that authentication updates last_login_at"""
        alice_id = self.um.username_index["alice"]
        user_before = self.um.get_user(alice_id)
        self.assertIsNone(user_before.last_login_at)
        
        self.um.authenticate("alice", "Password123")
        user_after = self.um.get_user(alice_id)
        self.assertIsNotNone(user_after.last_login_at)
    
    def test_brute_force_protection(self):
        """Test brute force attack protection"""
        alice_id = self.um.username_index["alice"]
        
        # Simulate 6 failed login attempts
        for i in range(6):
            self.um.authenticate("alice", "WrongPassword")
        
        # Next attempt should be blocked
        success, msg, user = self.um.authenticate("alice", "Password123")
        self.assertFalse(success)
        self.assertIn("Too many failed login attempts", msg)


class TestSessions(unittest.TestCase):
    """Test session management"""
    
    def setUp(self):
        self.um = UserManager()
        success, _, self.user = self.um.create_user("alice", "alice@example.com", "password123")
        self.user_id = self.user.user_id
    
    def test_create_session(self):
        """Test session creation"""
        session_id, session = self.um.create_session(self.user_id)
        self.assertIsNotNone(session_id)
        self.assertEqual(session.user_id, self.user_id)
        self.assertTrue(session.is_active)
    
    def test_validate_session_valid(self):
        """Test validating a valid session"""
        session_id, _ = self.um.create_session(self.user_id)
        is_valid, user_id = self.um.validate_session(session_id)
        self.assertTrue(is_valid)
        self.assertEqual(user_id, self.user_id)
    
    def test_validate_session_nonexistent(self):
        """Test validating nonexistent session"""
        is_valid, user_id = self.um.validate_session("fake-session-id")
        self.assertFalse(is_valid)
        self.assertIsNotNone(user_id)  # Returns error message, not None
    
    def test_revoke_session(self):
        """Test revoking a session"""
        session_id, _ = self.um.create_session(self.user_id)
        self.um.revoke_session(session_id)
        is_valid, _ = self.um.validate_session(session_id)
        self.assertFalse(is_valid)
    
    def test_revoke_all_user_sessions(self):
        """Test revoking all sessions for a user"""
        session_id1, _ = self.um.create_session(self.user_id)
        session_id2, _ = self.um.create_session(self.user_id)
        
        count = self.um.revoke_all_user_sessions(self.user_id)
        self.assertEqual(count, 2)
        
        is_valid1, _ = self.um.validate_session(session_id1)
        is_valid2, _ = self.um.validate_session(session_id2)
        self.assertFalse(is_valid1)
        self.assertFalse(is_valid2)


class TestAPITokens(unittest.TestCase):
    """Test API token management"""
    
    def setUp(self):
        self.um = UserManager()
        success, _, self.user = self.um.create_user("alice", "alice@example.com", "password123")
        self.user_id = self.user.user_id
    
    def test_create_api_key(self):
        """Test creating an API key"""
        token, token_obj = self.um.token_manager.create_token(
            self.user_id,
            token_type=TokenType.API_KEY
        )
        self.assertIsNotNone(token)
        self.assertEqual(token_obj.user_id, self.user_id)
        self.assertEqual(token_obj.token_type, TokenType.API_KEY)
    
    def test_create_session_token(self):
        """Test creating a session token"""
        token, token_obj = self.um.token_manager.create_token(
            self.user_id,
            token_type=TokenType.SESSION_TOKEN
        )
        # Session tokens should expire in 1 hour
        self.assertIsNotNone(token_obj.expires_at)
    
    def test_validate_token_valid(self):
        """Test validating a valid token"""
        token, _ = self.um.token_manager.create_token(self.user_id)
        is_valid, user_id, error = self.um.token_manager.validate_token(token)
        self.assertTrue(is_valid)
        self.assertEqual(user_id, self.user_id)
    
    def test_validate_token_invalid(self):
        """Test validating an invalid token"""
        is_valid, user_id, error = self.um.token_manager.validate_token("invalid-token")
        self.assertFalse(is_valid)
        self.assertIsNone(user_id)
        self.assertIn("not found", error)
    
    def test_revoke_token(self):
        """Test revoking a token"""
        token, token_obj = self.um.token_manager.create_token(self.user_id)
        token_hash = token_obj.token_hash
        
        self.um.token_manager.revoke_token(token_hash)
        is_valid, _, _ = self.um.token_manager.validate_token(token)
        self.assertFalse(is_valid)
    
    def test_revoke_user_tokens(self):
        """Test revoking all tokens for a user"""
        token1, t1 = self.um.token_manager.create_token(self.user_id)
        token2, t2 = self.um.token_manager.create_token(self.user_id)
        
        count = self.um.token_manager.revoke_user_tokens(self.user_id)
        self.assertEqual(count, 2)
    
    def test_token_with_scopes(self):
        """Test token with specific scopes"""
        token, token_obj = self.um.token_manager.create_token(
            self.user_id,
            scopes=["read", "write"]
        )
        self.assertEqual(token_obj.scopes, ["read", "write"])
    
    def test_token_with_ip_restrictions(self):
        """Test token with IP restrictions"""
        token, token_obj = self.um.token_manager.create_token(
            self.user_id,
            ip_restrictions=["192.168.1.1", "192.168.1.2"]
        )
        self.assertEqual(len(token_obj.ip_restrictions), 2)


class TestRolesAndPermissions(unittest.TestCase):
    """Test role and permission management"""
    
    def setUp(self):
        self.um = UserManager()
        success, _, self.user = self.um.create_user("alice", "alice@example.com", "password123")
        self.user_id = self.user.user_id
    
    def test_default_roles_exist(self):
        """Test that default roles are created"""
        roles = self.um.role_manager.list_roles()
        self.assertEqual(len(roles), 5)
        self.assertIn(UserRole.ADMIN.value, roles)
        self.assertIn(UserRole.DEVELOPER.value, roles)
        self.assertIn(UserRole.ANALYST.value, roles)
        self.assertIn(UserRole.VIEWER.value, roles)
        self.assertIn(UserRole.SERVICE_ACCOUNT.value, roles)
    
    def test_assign_role(self):
        """Test assigning a role to a user"""
        self.um.assign_role(self.user_id, UserRole.ADMIN.value)
        self.assertIn(UserRole.ADMIN.value, self.um.user_roles[self.user_id])
    
    def test_remove_role(self):
        """Test removing a role from a user"""
        self.um.assign_role(self.user_id, UserRole.ADMIN.value)
        self.um.remove_role(self.user_id, UserRole.ADMIN.value)
        self.assertNotIn(UserRole.ADMIN.value, self.um.user_roles[self.user_id])
    
    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions"""
        self.um.assign_role(self.user_id, UserRole.ADMIN.value)
        
        can_manage_users = self.um.can_user_perform_action(
            self.user_id, PermissionAction.MANAGE_USERS, "*"
        )
        self.assertTrue(can_manage_users)
        
        can_configure = self.um.can_user_perform_action(
            self.user_id, PermissionAction.CONFIGURE_SYSTEM, "*"
        )
        self.assertTrue(can_configure)
    
    def test_developer_has_api_access(self):
        """Test that developer role has API access"""
        self.um.assign_role(self.user_id, UserRole.DEVELOPER.value)
        
        can_read_api = self.um.can_user_perform_action(
            self.user_id, PermissionAction.READ, "api"
        )
        self.assertTrue(can_read_api)
        
        can_manage_users = self.um.can_user_perform_action(
            self.user_id, PermissionAction.MANAGE_USERS, "*"
        )
        self.assertFalse(can_manage_users)
    
    def test_analyst_can_view_analytics(self):
        """Test that analyst role can view analytics"""
        self.um.assign_role(self.user_id, UserRole.ANALYST.value)
        
        can_view = self.um.can_user_perform_action(
            self.user_id, PermissionAction.VIEW_ANALYTICS, "all"
        )
        self.assertTrue(can_view)
    
    def test_viewer_has_limited_access(self):
        """Test that viewer role has limited access"""
        # User already has viewer role by default
        can_read_analytics = self.um.can_user_perform_action(
            self.user_id, PermissionAction.READ, "analytics"
        )
        self.assertTrue(can_read_analytics)
        
        can_write = self.um.can_user_perform_action(
            self.user_id, PermissionAction.WRITE, "api"
        )
        self.assertFalse(can_write)
    
    def test_multiple_roles(self):
        """Test user with multiple roles"""
        self.um.assign_role(self.user_id, UserRole.DEVELOPER.value)
        self.um.assign_role(self.user_id, UserRole.ANALYST.value)
        
        roles = self.um.user_roles[self.user_id]
        self.assertEqual(len(roles), 3)  # viewer + developer + analyst
    
    def test_get_user_permissions(self):
        """Test getting all permissions for a user"""
        self.um.assign_role(self.user_id, UserRole.ADMIN.value)
        permissions = self.um.get_user_permissions(self.user_id)
        self.assertGreater(len(permissions), 0)


class TestAuditLogging(unittest.TestCase):
    """Test audit logging functionality"""
    
    def setUp(self):
        self.um = UserManager()
        success, _, self.user = self.um.create_user("alice", "alice@example.com", "password123")
        self.user_id = self.user.user_id
    
    def test_user_creation_logged(self):
        """Test that user creation is logged"""
        entries = self.um.audit_logger.get_user_activity(self.user_id)
        creation_entries = [e for e in entries if e["action"] == AuditAction.USER_CREATED.value]
        self.assertGreater(len(creation_entries), 0)
    
    def test_login_success_logged(self):
        """Test that successful login is logged"""
        self.um.authenticate("alice", "password123")
        entries = self.um.audit_logger.get_user_activity(self.user_id)
        login_entries = [e for e in entries if e["action"] == AuditAction.LOGIN_SUCCESS.value]
        self.assertGreater(len(login_entries), 0)
    
    def test_login_failure_logged(self):
        """Test that failed login is logged"""
        self.um.authenticate("alice", "wrongpassword")
        entries = self.um.audit_logger.get_all_entries()
        failure_entries = [e for e in entries if e["action"] == AuditAction.LOGIN_FAILED.value]
        self.assertGreater(len(failure_entries), 0)
    
    def test_role_assignment_logged(self):
        """Test that role assignment is logged"""
        self.um.assign_role(self.user_id, UserRole.ADMIN.value)
        entries = self.um.audit_logger.get_user_activity(self.user_id)
        role_entries = [e for e in entries if e["action"] == AuditAction.ROLE_ASSIGNED.value]
        self.assertGreater(len(role_entries), 0)
    
    def test_get_failed_login_attempts(self):
        """Test getting failed login attempts"""
        for _ in range(3):
            self.um.authenticate("alice", "wrongpassword")
        
        count = self.um.audit_logger.get_failed_login_attempts(self.user_id, hours=1)
        self.assertEqual(count, 3)


class TestUserManagement(unittest.TestCase):
    """Test general user management operations"""
    
    def setUp(self):
        self.um = UserManager()
    
    def test_get_user_info(self):
        """Test getting comprehensive user information"""
        self.um.create_user("alice", "alice@example.com", "password123")
        alice_id = self.um.username_index["alice"]
        
        info = self.um.get_user_info(alice_id)
        self.assertIsNotNone(info)
        self.assertEqual(info["profile"]["username"], "alice")
        self.assertIn("roles", info)
        self.assertIn("permissions", info)
    
    def test_update_user(self):
        """Test updating user information"""
        self.um.create_user("alice", "alice@example.com", "password123")
        alice_id = self.um.username_index["alice"]
        
        self.um.update_user(alice_id, full_name="Alice Admin", organization="ACME Corp")
        user = self.um.get_user(alice_id)
        self.assertEqual(user.full_name, "Alice Admin")
        self.assertEqual(user.organization, "ACME Corp")
    
    def test_delete_user(self):
        """Test deleting a user"""
        self.um.create_user("alice", "alice@example.com", "password123")
        alice_id = self.um.username_index["alice"]
        
        self.um.delete_user(alice_id)
        self.assertIsNone(self.um.get_user(alice_id))
        self.assertNotIn("alice", self.um.username_index)
    
    def test_get_all_users(self):
        """Test getting all users"""
        self.um.create_user("alice", "alice@example.com", "password123")
        self.um.create_user("bob", "bob@example.com", "password123")
        
        users = self.um.get_all_users()
        self.assertEqual(len(users), 2)
    
    def test_system_stats(self):
        """Test system statistics"""
        self.um.create_user("alice", "alice@example.com", "password123")
        alice_id = self.um.username_index["alice"]
        self.um.create_session(alice_id)
        
        stats = self.um.get_system_stats()
        self.assertEqual(stats["total_users"], 1)
        self.assertEqual(stats["active_users"], 1)
        self.assertGreater(stats["total_sessions"], 0)


class TestOrchestrator(unittest.TestCase):
    """Test the orchestrator"""
    
    def setUp(self):
        self.orchestrator = UserAuthOrchestrator()
    
    def test_initialize(self):
        """Test orchestrator initialization"""
        result = self.orchestrator.initialize()
        self.assertEqual(result["status"], "initialized")
    
    def test_system_report(self):
        """Test getting system report"""
        report = self.orchestrator.get_system_report()
        self.assertIn("system", report)
        self.assertIn("roles", report)


# Run all tests
if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 16: USER MANAGEMENT & AUTHENTICATION - TEST SUITE")
    print("=" * 80)
    print()
    
    unittest.main(verbosity=2)
