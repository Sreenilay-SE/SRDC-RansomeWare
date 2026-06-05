import os
from PIL import Image, ImageDraw

def draw_shield(size):
    # Create image with transparent background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Shield shape points relative to size
    margin = size * 0.15
    top_y = size * 0.15
    mid_y = size * 0.45
    bot_y = size * 0.85
    left_x = margin
    right_x = size - margin
    center_x = size / 2
    
    points = [
        (center_x, top_y),        # Top Center
        (right_x, top_y),         # Top Right
        (right_x, mid_y),         # Mid Right
        (center_x, bot_y),        # Bottom Center (Tip)
        (left_x, mid_y),          # Mid Left
        (left_x, top_y)           # Top Left
    ]
    
    # Draw dark glowing background shield
    draw.polygon(points, fill=(15, 23, 42, 235), outline=(59, 130, 246, 255), width=max(1, int(size * 0.05)))
    
    # Draw internal shield highlight (inner glowing line)
    inner_margin = size * 0.22
    inner_top_y = size * 0.22
    inner_mid_y = size * 0.45
    inner_bot_y = size * 0.78
    inner_left_x = inner_margin
    inner_right_x = size - inner_margin
    
    inner_points = [
        (center_x, inner_top_y),
        (inner_right_x, inner_top_y),
        (inner_right_x, inner_mid_y),
        (center_x, inner_bot_y),
        (inner_left_x, inner_mid_y),
        (inner_left_x, inner_top_y)
    ]
    draw.polygon(inner_points, fill=(59, 130, 246, 45), outline=(96, 165, 250, 180), width=max(1, int(size * 0.02)))
    
    return img

def main():
    icons_dir = "icons"
    os.makedirs(icons_dir, exist_ok=True)
    
    sizes = [16, 48, 128]
    for size in sizes:
        img = draw_shield(size)
        path = os.path.join(icons_dir, f"shield_{size}.png")
        img.save(path, "PNG")
        print(f"[SUCCESS] Saved extension icon: {path} ({size}x{size})")

if __name__ == "__main__":
    main()
