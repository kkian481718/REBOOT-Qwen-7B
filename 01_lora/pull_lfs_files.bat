@echo off
REM 從 Git LFS 下載實際的模型檔案

echo ====================================
echo Git LFS 檔案下載工具
echo ====================================
echo.

REM 檢查是否安裝 git-lfs
git lfs version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 錯誤: 未安裝 Git LFS
    echo.
    echo 💡 解決方案:
    echo    1. 下載 Git LFS: https://git-lfs.github.com/
    echo    2. 安裝後執行: git lfs install
    echo    3. 重新執行此腳本
    echo.
    pause
    exit /b 1
)

echo ✅ Git LFS 已安裝
echo.

REM 切換到專案根目錄
cd /d "%~dp0\.."

echo 📥 正在下載 LoRA 模型檔案...
echo.

git lfs pull

if %errorlevel% equ 0 (
    echo.
    echo ====================================
    echo ✅ 下載完成！
    echo ====================================
) else (
    echo.
    echo ====================================
    echo ❌ 下載失敗
    echo ====================================
    echo.
    echo 💡 可能的原因：
    echo    1. 沒有連接到 Git 遠端倉庫
    echo    2. 網路連線問題
    echo    3. 沒有權限存取 LFS 檔案
    echo.
    echo 💡 替代方案：
    echo    1. 重新訓練模型
    echo    2. 從其他來源取得完整的 LoRA 檔案
    echo    3. 暫時使用基礎模型（不載入 LoRA）
)

echo.
pause
