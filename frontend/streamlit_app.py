"""Streamlit 调试前端。

该页面主要用于本地联调后端接口，包含：
1. 健康检查；
2. 文档上传与索引重建；
3. 问答与历史查询。
"""

import uuid
from typing import Any

import httpx
import streamlit as st

# 页面基础配置。
st.set_page_config(page_title="RAG 助手", layout="wide")
st.title("RAG 知识助手")

# 首次进入页面时自动生成会话 ID。
if "session_id" not in st.session_state:
    st.session_state.session_id = f"s-{uuid.uuid4().hex[:8]}"


def _to_json(resp: httpx.Response) -> dict[str, Any]:
    """把 HTTP 响应安全转换为可展示字典。

    Args:
        resp: httpx 响应对象。

    Returns:
        dict[str, Any]: 解析成功返回 JSON；失败返回原始文本。
    """
    try:
        data = resp.json()
        if isinstance(data, dict):
            return data
        return {"raw": data}
    except Exception:
        return {"raw_text": resp.text}


def _api_get(api_base: str, path: str, timeout: float) -> tuple[int, dict[str, Any]]:
    """发送 GET 请求。

    Args:
        api_base: 后端基地址。
        path: 接口路径。
        timeout: 超时时间（秒）。

    Returns:
        tuple[int, dict[str, Any]]: `(HTTP 状态码, 响应体)`。
    """
    resp = httpx.get(f"{api_base}{path}", timeout=timeout)
    return resp.status_code, _to_json(resp)


def _api_post_json(
    api_base: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """发送 JSON POST 请求。"""
    resp = httpx.post(f"{api_base}{path}", json=payload, timeout=timeout)
    return resp.status_code, _to_json(resp)


def _api_post_file(
    api_base: str,
    path: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """发送文件上传 POST 请求。"""
    files = {"file": (file_name, file_bytes, content_type)}
    resp = httpx.post(f"{api_base}{path}", files=files, timeout=timeout)
    return resp.status_code, _to_json(resp)


# 后端地址输入框。
api_base = st.text_input("后端 API 地址", value="http://localhost:8000")
# 超时时间滑块。
timeout_seconds = st.slider("请求超时（秒）", min_value=3, max_value=60, value=20)

# 三个功能标签页。
tab_health, tab_upload, tab_chat = st.tabs(["健康检查", "上传与建索引", "问答与历史"])

with tab_health:
    # 点击后触发健康检查请求。
    if st.button("检查服务状态", use_container_width=True):
        try:
            status_code, body = _api_get(api_base, "/api/v1/health", float(timeout_seconds))
            st.success(f"HTTP {status_code}")
            st.json(body)
        except Exception as exc:
            st.error(f"请求失败：{exc}")

with tab_upload:
    # 文件上传组件。
    uploaded_file = st.file_uploader(
        "上传知识文档（txt/md/pdf/docx）",
        type=["txt", "md", "pdf", "docx"],
    )

    # 两列按钮：左上传并重建，右仅重建。
    col_upload, col_rebuild = st.columns(2)
    with col_upload:
        if st.button("上传并重建索引", use_container_width=True):
            if uploaded_file is None:
                st.warning("请先选择一个文件。")
            else:
                try:
                    content_type = uploaded_file.type or "application/octet-stream"
                    status_code, body = _api_post_file(
                        api_base=api_base,
                        path="/api/v1/upload",
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        content_type=content_type,
                        timeout=float(timeout_seconds),
                    )
                    if status_code == 200:
                        st.success("上传并重建成功。")
                    else:
                        st.error(f"上传失败，HTTP {status_code}")
                    st.json(body)
                except Exception as exc:
                    st.error(f"请求失败：{exc}")

    with col_rebuild:
        if st.button("仅重建索引", use_container_width=True):
            try:
                status_code, body = _api_post_json(
                    api_base=api_base,
                    path="/api/v1/index/rebuild",
                    payload={},
                    timeout=float(timeout_seconds),
                )
                if status_code == 200:
                    st.success("索引重建完成。")
                else:
                    st.error(f"重建失败，HTTP {status_code}")
                st.json(body)
            except Exception as exc:
                st.error(f"请求失败：{exc}")

with tab_chat:
    # 会话 ID 输入框（默认用 session_state 中自动生成值）。
    session_id = st.text_input("会话 ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id
    # 可选来源过滤。
    source_filter = st.text_input("来源过滤（可选）", value="")
    # 问题输入框。
    question = st.text_area("问题", placeholder="例如：这份文档讲了哪些关键点？")

    if st.button("发送问题", use_container_width=True):
        if not question.strip():
            st.warning("请输入问题。")
        else:
            payload: dict[str, Any] = {
                "session_id": st.session_state.session_id,
                "question": question.strip(),
            }
            if source_filter.strip():
                payload["source_filter"] = source_filter.strip()

            try:
                status_code, body = _api_post_json(
                    api_base=api_base,
                    path="/api/v1/chat",
                    payload=payload,
                    timeout=float(timeout_seconds),
                )
                if status_code == 200:
                    st.success("问答成功。")
                else:
                    st.error(f"问答失败，HTTP {status_code}")
                st.json(body)
            except Exception as exc:
                st.error(f"请求失败：{exc}")

    if st.button("加载会话历史", use_container_width=True):
        try:
            status_code, body = _api_get(
                api_base=api_base,
                path=f"/api/v1/history/{st.session_state.session_id}",
                timeout=float(timeout_seconds),
            )
            if status_code == 200:
                st.success("历史加载成功。")
            else:
                st.error(f"加载失败，HTTP {status_code}")
            st.json(body)
        except Exception as exc:
            st.error(f"请求失败：{exc}")
