from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
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
from databases import (
    get_db, SessionLocal, UserLogin, 
    create_password_reset_token, verify_password_reset_token, 
    use_password_reset_token, cleanup_expired_tokens
)
from sqlalchemy.orm import Session
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 配置日誌
logger = logging.getLogger(__name__)

# 寄信設定
SMTP_HOST = "smtp.your-email.com" #要連到哪台 SMTP（寄信）伺服器
SMTP_PORT = 587 #寄信用的通訊埠587 是 STARTTLS（先用不加密連線再升級到 TLS）465 通常是直接用 SSL 加密
SMTP_USER = "noreply@your-domain.com" #要用哪個郵箱地址當「寄件人」
SMTP_PASS = "your-smtp-password" #應用程式專用密碼（App Password）或「授權碼」
FRONTEND_URL = "https://your-domain.com"  # 網頁前端網址

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

def send_reset_email(email: str, token: str, background_tasks: BackgroundTasks):
    """发送密码重设邮件"""
    def send_email():
        try:
            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = FRONTEND_URL
            msg['To'] = email
            msg['Subject'] = "密碼重設請求"
            
            # 重設連結（根據前端地址修改）
            reset_link = f"http://localhost:3000/reset-password?token={token}"
            
            # 郵件本文
            body = f"""
            <html>
            <body>
                <h2>密碼重設請求</h2>
                <p>您收到此信件是因为有人请求重设您的账户密码。</p>
                <p>如果这是您的操作，请点击下面的链接重设密码：</p>
                <p><a href="{reset_link}">重设密码</a></p>
                <p>此链接将在1小时后过期。</p>
                <p>如果您没有请求重设密码，请忽略此邮件。</p>
                <br>
                <p>注意：为了您的账户安全，请不要将此链接分享给他人。</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # 发送邮件
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            text = msg.as_string()
            server.sendmail(FRONTEND_URL, email, text)
            server.quit()
            
            logger.info(f"重設密碼郵件成功送出: {email}")
            
        except Exception as e:
            logger.error(f"郵件發送失敗: {e}")
    
    # 將發送郵件添加到後台執行
    background_tasks.add_task(send_email)

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
    #print("收到註冊資料：", request.dict())
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
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
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
        token = create_password_reset_token(request.email)
        
        if token:
            # 发送重设邮件
            send_reset_email(request.email, token, background_tasks)
            
            logger.info(f"密碼重設 token 生成成功: {user.email}")
            
            return MessageResponse(
                message="密碼重設連結已發送到您的信箱",
                success=True
            )
        else:
            raise Exception("生成重设令牌失败")
        
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
    
    # 驗證token
    email = verify_password_reset_token(request.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="重設連結已經過期，請重新申請重設密碼"
        )
    
    try:
        # 查找用戶
        user = db.query(UserLogin).filter(UserLogin.email == email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用戶不存在"
            )
        
        # 更新密碼
        user.password = get_password_hash(request.new_password)
        
        # 標記token為已使用
        use_password_reset_token(request.token)
        
        # 更新登入狀態為未登入（強制重新登入）
        user.login_stat = False
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"用戶密碼重設成功: {user.email}")
        
        return MessageResponse(
            message="密碼重設成功！請使用新密碼登入",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"重設密碼時發生錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="重設密碼時發生錯誤，請稍後再試"
        )

@auth_router.get("/verify-reset-token/{token}")
async def verify_reset_token(token: str):
    """驗證重設token是否有效"""
    email = verify_password_reset_token(token)
    
    if email:
        return {
            "valid": True,
            "message": "驗證有效，可以重設密碼"
        }
    else:
        return {
            "valid": False,
            "message": "驗證無效或已過期"
        }

# 定期清理過期token的後台任務
@auth_router.post("/dev/cleanup-expired-tokens", response_model=MessageResponse)
async def cleanup_expired_tokens_endpoint():
    """清理過期的密碼重設token（管理員功能）"""
    try:
        cleanup_expired_tokens()
        return MessageResponse(
            message="過期token清理完成",
            success=True
        )
    except Exception as e:
        logger.error(f"清理過期token時發生錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清理過期token失誤"
        )

# 如果不使用真实邮件服务，可以使用这个开发版本
@auth_router.post("/dev/forgot-password", response_model=dict)
async def dev_forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """开发环境的忘记密码（直接返回令牌，不发送邮件）"""
    logger.info(f"[开发模式] 使用者请求密码重设: {request.email}")
    
    # 检查邮箱是否存在
    user = db.query(UserLogin).filter(UserLogin.email == request.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="邮箱不存在"
        )
    
    try:
        # 创建重设令牌
        token = create_password_reset_token(request.email)
        
        if token:
            # 在开发环境中直接返回令牌
            return {
                "message": "密码重设令牌生成成功（开发模式）",
                "success": True,
                "token": token,  # 生产环境中不应该返回令牌
                "reset_url": f"http://localhost:3000/reset-password?token={token}"
            }
        else:
            raise Exception("生成重设令牌失败")
            
    except Exception as e:
        logger.error(f"密码重设时发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="处理密码重设时发生错误"
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