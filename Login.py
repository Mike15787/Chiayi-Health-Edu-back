from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import secrets
import hashlib
import logging
import uuid
from typing import Optional, Dict, Any
from databases import get_db, SessionLocal, UserLogin
from sqlalchemy.orm import Session

# 配置日誌
logger = logging.getLogger(__name__)

# 安全配置
SECRET_KEY = secrets.token_urlsafe(32)  # 在生產環境中應從環境變量讀取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密碼加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer Token
security = HTTPBearer()

# 創建路由器
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

# --- Pydantic 模型 ---
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    confirm_password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_info: Dict[str, Any]

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    uid: str
    created_at: datetime
    login_stat: bool

class MessageResponse(BaseModel):
    message: str
    success: bool

# --- 工具函數 ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """驗證密碼"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """生成密碼雜湊"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """創建 JWT Token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """驗證 JWT Token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError:
        return None

def authenticate_user(db: Session, username: str, password: str) -> Optional[UserLogin]:
    """驗證使用者帳號密碼"""
    try:
        # 查詢使用者 (支援用戶名或郵箱登入)
        user = db.query(UserLogin).filter(
            (UserLogin.username == username) | (UserLogin.email == username)
        ).first()
        
        if not user:
            return None
        
        if not verify_password(password, user.password):
            return None
        
        return user
    except Exception as e:
        logger.error(f"認證使用者時發生錯誤: {e}")
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> UserLogin:
    """獲取當前使用者"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    user = db.query(UserLogin).filter(UserLogin.username == username).first()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def update_login_status(db: Session, user: UserLogin, status: bool):
    """更新用戶登入狀態"""
    try:
        user.login_stat = status
        db.commit()
        db.refresh(user)
    except Exception as e:
        logger.error(f"更新登入狀態時發生錯誤: {e}")
        db.rollback()

# --- API 端點 ---
@auth_router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """使用者登入"""
    logger.info(f"使用者嘗試登入: {request.username}")
    
    # 驗證使用者
    user = authenticate_user(db, request.username, request.password)
    if not user:
        logger.warning(f"登入失敗: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="帳號或密碼錯誤",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 更新登入狀態
    update_login_status(db, user, True)
    
    # 創建 access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "uid": user.uid},
        expires_delta=access_token_expires
    )
    
    logger.info(f"使用者登入成功: {user.username}")
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_info={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "uid": user.uid,
            "login_stat": user.login_stat
        }
    )

@auth_router.post("/register", response_model=MessageResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """使用者註冊"""
    print("收到註冊資料：", request.dict())
    logger.info(f"使用者嘗試註冊: {request.username}")
    
    # 驗證密碼確認
    if request.password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密碼確認不一致"
        )
    
    # 檢查使用者名稱是否已存在
    existing_user = db.query(UserLogin).filter(UserLogin.username == request.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="使用者名稱已存在"
        )
    
    # 檢查郵箱是否已存在
    existing_email = db.query(UserLogin).filter(UserLogin.email == request.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="郵箱已被註冊"
        )
    
    # 創建新使用者
    try:
        password_hash = get_password_hash(request.password)
        user_uid = str(uuid.uuid4())  # 生成唯一 UUID
        
        new_user = UserLogin(
            username=request.username,
            email=request.email,
            password=password_hash,
            uid=user_uid,
            login_stat=False  # 註冊後預設未登入
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"使用者註冊成功: {request.username}")
        
        return MessageResponse(
            message="註冊成功！請使用帳號密碼登入",
            success=True
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"註冊時發生錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="註冊時發生錯誤，請稍後再試"
        )

@auth_router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """忘記密碼"""
    logger.info(f"使用者請求密碼重設: {request.email}")
    
    # 檢查郵箱是否存在
    user = db.query(UserLogin).filter(UserLogin.email == request.email).first()
    if not user:
        # 為了安全起見，不透露郵箱是否存在
        logger.warning(f"密碼重設請求：郵箱不存在 {request.email}")
        return MessageResponse(
            message="如果該郵箱已註冊，重設連結已發送到您的信箱",
            success=True
        )
    
    try:
        # 生成重設 token (這裡簡化處理，實際應用中需要存儲到資料庫)
        reset_token = secrets.token_urlsafe(32)
        
        # TODO: 在實際應用中，需要：
        # 1. 將 token 存儲到資料庫並設置過期時間
        # 2. 發送包含 token 的重設連結到用戶郵箱
        
        logger.info(f"密碼重設 token 生成成功: {user.email}")
        
        return MessageResponse(
            message="密碼重設連結已發送到您的信箱",
            success=True
        )
        
    except Exception as e:
        logger.error(f"密碼重設時發生錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="處理密碼重設時發生錯誤"
        )

@auth_router.post("/reset-password", response_model=MessageResponse)
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """重設密碼"""
    logger.info("使用者嘗試重設密碼")
    
    # 驗證密碼確認
    if request.new_password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密碼確認不一致"
        )
    
    # TODO: 在實際應用中，需要：
    # 1. 驗證 token 的有效性和過期時間
    # 2. 從資料庫中查找對應的使用者
    # 3. 更新使用者密碼
    
    return MessageResponse(
        message="密碼重設成功！請使用新密碼登入",
        success=True
    )

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserLogin = Depends(get_current_user)):
    """獲取當前使用者資訊"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        uid=current_user.uid,
        created_at=current_user.created_at,
        login_stat=current_user.login_stat
    )

@auth_router.post("/logout", response_model=MessageResponse)
async def logout(current_user: UserLogin = Depends(get_current_user), db: Session = Depends(get_db)):
    """使用者登出"""
    logger.info(f"使用者登出: {current_user.username}")
    
    # 更新登入狀態
    update_login_status(db, current_user, False)
    
    # TODO: 在實際應用中，可以：
    # 1. 將 token 加入黑名單
    # 2. 清除 session 資料
    
    return MessageResponse(
        message="登出成功",
        success=True
    )

@auth_router.get("/verify-token")
async def verify_token_endpoint(current_user: UserLogin = Depends(get_current_user)):
    """驗證 token 有效性"""
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "uid": current_user.uid,
            "login_stat": current_user.login_stat
        }
    }

# --- 開發用的測試端點 ---
@auth_router.post("/dev/create-test-users", response_model=MessageResponse)
async def create_test_users(db: Session = Depends(get_db)):
    """創建測試使用者（僅開發環境使用）"""
    try:
        # 創建管理員使用者
        admin_user = UserLogin(
            username="admin",
            email="admin@example.com",
            password=get_password_hash("password"),
            uid=str(uuid.uuid4()),
            login_stat=False
        )
        
        # 創建一般使用者
        normal_user = UserLogin(
            username="user",
            email="user@example.com",
            password=get_password_hash("password"),
            uid=str(uuid.uuid4()),
            login_stat=False
        )
        
        # 檢查是否已存在
        existing_admin = db.query(UserLogin).filter(UserLogin.username == "admin").first()
        existing_user = db.query(UserLogin).filter(UserLogin.username == "user").first()
        
        if not existing_admin:
            db.add(admin_user)
        if not existing_user:
            db.add(normal_user)
        
        db.commit()
        
        return MessageResponse(
            message="測試使用者創建成功 (admin/password, user/password)",
            success=True
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"創建測試使用者時發生錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="創建測試使用者失敗"
        )