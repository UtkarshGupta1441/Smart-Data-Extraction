import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import os
import shutil
import stat
import time
from playwright.async_api import async_playwright
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output2")
DEFAULT_SCREENSHOT_DIR = os.path.join(BASE_DIR, "smart_screenshots")
MODEL_PATH = os.getenv(
    "DOCSEG_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "yolov11l-doclaynet.pt")
)
_docseg_model = None

# === YOLO model loading and image enhancement ===
def get_docseg_model():
    global _docseg_model
    if _docseg_model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"DocLayNet model not found at '{MODEL_PATH}'. "
                "Set DOCSEG_MODEL_PATH or place the model under models/."
            )
        _docseg_model = YOLO(MODEL_PATH)
    return _docseg_model

def resolve_path(path, base_dir=BASE_DIR):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)

def enhance_image(img):
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    return img

def get_label_color(label):
    color_mapping = {
        'table': 'blue',
        'picture': 'green',
        'text': 'red',
        'figure': 'orange',
        'page-header': 'purple',
        'section-header': 'cyan',
        'formula': 'black',
        'page-footer': 'magenta'
    }
    return color_mapping.get(label, 'red')

def process_image(image_path, output_base_dir=DEFAULT_OUTPUT_DIR):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_base_dir = resolve_path(output_base_dir)
    output_dir = os.path.join(output_base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    original_img = img.copy()
    img = enhance_image(img)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    model = get_docseg_model()
    results = model(source=np.array(img), conf=0.1, iou=0.5)
    label_counts = {}

    for box in results[0].boxes:
        bbox = box.xyxy[0].tolist()
        cls_index = int(box.cls.item())
        label = results[0].names[cls_index].lower()
        confidence = box.conf.item()

        #print(f"Model detected label: '{label}' -> mapped to color: {get_label_color(label)}")

        if label == 'list-item':
            label = 'text'

        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        x1, y1, x2, y2 = map(int, bbox)
        cropped_region = original_img.crop((x1, y1, x2, y2))
        label_counts[label] = label_counts.get(label, 0) + 1
        cropped_filename = f"{label}_{label_counts[label]}.png"
        cropped_region.save(os.path.join(label_dir, cropped_filename))

        color = get_label_color(label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label}: {confidence:.2f}"
        draw.text((x1, y1 - 25), text, fill=color, font=font)

    annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
    img.save(annotated_path)
    print(f"Processed image saved to: {annotated_path}")

def process_folder(folder_path, output_base_dir=DEFAULT_OUTPUT_DIR):
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_exts):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {image_path}")
            process_image(image_path, output_base_dir)

def clear_screenshot_folder(folder_path):
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_exts):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Warning: could not remove old screenshot {file_path}: {e}")

def _remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def clear_output2_folder(folder_path=DEFAULT_OUTPUT_DIR, retries=3, retry_delay=0.35):
    folder_path = resolve_path(folder_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return

    cleared = False
    for attempt in range(retries):
        try:
            shutil.rmtree(folder_path, onerror=_remove_readonly)
            cleared = True
            break
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                break
        except OSError:
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                break

    if not cleared and os.path.isdir(folder_path):
        stale_name = f"{folder_path}_stale_{int(time.time())}"
        try:
            os.rename(folder_path, stale_name)
            cleared = True
        except OSError as e:
            print(f"Warning: could not fully clear {folder_path}: {e}")

    os.makedirs(folder_path, exist_ok=True)

# === Playwright screenshot async function ===
async def take_smart_screenshots(url, output_dir=DEFAULT_SCREENSHOT_DIR, max_scrolls=500):
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    clear_screenshot_folder(output_dir)

    async def dismiss_popups(target_page):
        close_selectors = [
            '[aria-label="Close"]',
            '[aria-label="close"]',
            'button:has-text("Close")',
            'button:has-text("Dismiss")',
            'button:has-text("Not now")',
            'button:has-text("No thanks")',
            'button:has-text("Skip")',
            'button:has-text("×")',
            'button:has-text("✕")',
            '.close',
            '.modal .close',
            '.popup .close',
            'div[role="dialog"] button'
        ]

        for selector in close_selectors:
            try:
                locator = target_page.locator(selector)
                count = await locator.count()
                for i in range(min(count, 3)):
                    item = locator.nth(i)
                    if await item.is_visible():
                        try:
                            await item.click(timeout=500, force=True)
                            await target_page.wait_for_timeout(120)
                        except Exception:
                            pass
            except Exception:
                pass

        try:
            await target_page.keyboard.press("Escape")
        except Exception:
            pass

        try:
            await target_page.evaluate("""() => {
                const knownOverlaySelectors = [
                    '.overlay', '.modal', '.popup', '.interstitial',
                    '.pw-overlay', '.js-signupPrompt', '.js-meterInterstitial',
                    '[class*="overlay"]', '[class*="popup"]', '[class*="modal"]',
                    '[class*="paywall"]', '[class*="subscribe"]', '[class*="newsletter"]',
                    'div[role="dialog"]', 'iframe'
                ];

                for (const selector of knownOverlaySelectors) {
                    document.querySelectorAll(selector).forEach(el => {
                        el.style.setProperty('display', 'none', 'important');
                        el.style.setProperty('visibility', 'hidden', 'important');
                        el.style.setProperty('opacity', '0', 'important');
                        el.style.setProperty('pointer-events', 'none', 'important');
                    });
                }

                const adKeywords = ['patreon', 'ko-fi', 'kofi', 'liberapay', 'donate', 'milestone', 'support us'];
                const nodes = Array.from(document.querySelectorAll('div,section,aside'));
                nodes.forEach(node => {
                    const text = (node.innerText || '').toLowerCase();
                    const style = window.getComputedStyle(node);
                    const isFloating = ['fixed', 'sticky'].includes(style.position);
                    const highZ = parseInt(style.zIndex || '0', 10) >= 100;
                    const isAdLike = adKeywords.some(k => text.includes(k));
                    if (isFloating && (highZ || isAdLike)) {
                        node.style.setProperty('display', 'none', 'important');
                        node.style.setProperty('visibility', 'hidden', 'important');
                        node.style.setProperty('opacity', '0', 'important');
                        node.style.setProperty('pointer-events', 'none', 'important');
                    }
                });
            }""")
        except Exception:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            locale="en-US"
        )

        # Hide modals/popups
        await context.add_init_script("""
            const style = document.createElement('style');
            style.innerHTML = `
                [aria-label="Close"],
                .overlay,
                .modal,
                .popup,
                div[role="dialog"],
                iframe,
                .pw-overlay,
                .js-signupPrompt,
                .js-meterInterstitial,
                [class*="overlay"],
                [class*="popup"],
                [class*="modal"],
                [class*="paywall"] {
                    display: none !important;
                    visibility: hidden !important;
                    opacity: 0 !important;
                    pointer-events: none !important;
                    z-index: -1 !important;
                }
                body {
                    overflow: auto !important;
                }
            `;
            document.head.appendChild(style);
        """)

        # Manual stealth patch
        page = await context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        try:
            await page.goto(url, wait_until='load', timeout=60000)
            await page.wait_for_timeout(1200)
            await dismiss_popups(page)

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)
            total_height = await page.evaluate("document.body.scrollHeight")

            screenshot_count = 0
            last_scroll_y = 0
            last_capture_y = -1
            min_end_margin = 5

            for _ in range(max_scrolls):
                await page.evaluate(f"window.scrollTo(0, {last_scroll_y})")
                await page.wait_for_timeout(700)

                current_y = await page.evaluate("window.scrollY")
                current_total_height = await page.evaluate("document.body.scrollHeight")
                viewport_height = await page.evaluate("window.innerHeight")
                page_bottom_y = max(current_total_height - viewport_height, 0)
                capture_step = max(100, int(viewport_height * 0.22))
                max_scroll_step = max(110, int(viewport_height * 0.28))
                near_end = current_y >= page_bottom_y - min_end_margin

                await dismiss_popups(page)
                await page.wait_for_timeout(120)

                should_capture = (
                    screenshot_count == 0
                    or (current_y - last_capture_y) >= capture_step
                    or near_end
                )

                if should_capture:
                    screenshot_path = os.path.join(output_dir, f"screenshot_{screenshot_count + 1}.png")
                    await page.screenshot(path=screenshot_path, full_page=False)
                    print(f"Saved: {screenshot_path}")
                    screenshot_count += 1
                    last_capture_y = current_y

                if near_end:
                    break

                next_scroll_y = await page.evaluate("""() => {
                    const currentY = window.scrollY;
                    const viewHeight = window.innerHeight;
                    const elements = Array.from(document.querySelectorAll('article, section, div, img, table, p'));

                    const candidates = elements.map(el => {
                        const rect = el.getBoundingClientRect();
                        return {
                            top: rect.top + window.scrollY,
                            bottom: rect.bottom + window.scrollY,
                            height: rect.height
                        };
                    }).filter(el =>
                        el.top > currentY + 20 &&
                        el.bottom - currentY > viewHeight * 0.3
                    );

                    candidates.sort((a, b) => a.top - b.top);
                    return candidates.length ? candidates[0].top : -1;
                }""")

                if next_scroll_y == -1:
                    fallback_y = min(current_y + max_scroll_step, page_bottom_y)
                    if fallback_y <= current_y + 5:
                        break
                    last_scroll_y = fallback_y
                    continue

                scroll_delta = next_scroll_y - current_y
                if scroll_delta <= 5:
                    fallback_y = min(current_y + max_scroll_step, page_bottom_y)
                    if fallback_y <= current_y + 5:
                        break
                    last_scroll_y = fallback_y
                    continue

                bounded_next_scroll_y = min(next_scroll_y, current_y + max_scroll_step, page_bottom_y)
                if bounded_next_scroll_y <= current_y + 5:
                    fallback_y = min(current_y + max_scroll_step, page_bottom_y)
                    if fallback_y <= current_y + 5:
                        break
                    last_scroll_y = fallback_y
                    continue

                last_scroll_y = bounded_next_scroll_y
                total_height = max(total_height, current_total_height)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

    return output_dir  # Return screenshots folder path

def run_pipeline(url, output_dir=DEFAULT_OUTPUT_DIR, screenshots_dir=DEFAULT_SCREENSHOT_DIR):
    output_dir = resolve_path(output_dir)
    screenshots_dir = resolve_path(screenshots_dir)

    clear_output2_folder(output_dir)

    screenshots_folder = asyncio.run(
        take_smart_screenshots(url, output_dir=screenshots_dir)
    )

    print(f"Processing screenshots in folder: {screenshots_folder}")
    process_folder(screenshots_folder, output_base_dir=output_dir)
    return output_dir

# === Main ===
if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://freedium-mirror.cfd/fa1c7992b195"
    run_pipeline(url)
