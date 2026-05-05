
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Dữ liệu đầu vào (tên sản phẩm, đơn giá, số lượng)
data = [
    {"name": "Laptop Gaming Dell G15", "price": 25000000, "qty": 1},
    {"name": "Chuột Logitech B100", "price": 100000, "qty": 50},
    {"name": "Màn hình LG UltraWide", "price": 8000000, "qty": 3},
    {"name": "Cáp chuyển đổi USB-C", "price": 150000, "qty": 10},
    {"name": "Server HP ProLiant", "price": 60000000, "qty": 1},
    {"name": "Bàn phím cơ Keychron", "price": 2000000, "qty": 5},
    {"name": "Tai nghe Sony WH-1000XM5", "price": 7000000, "qty": 4},
    {"name": "Ổ cứng HDD WD Red 4TB", "price": 3000000, "qty": 10},
    {"name": "Lót chuột khổ lớn", "price": 50000, "qty": 20},
    {"name": "Macbook Pro M3", "price": 45000000, "qty": 2},
    {"name": "USB Kingston 64GB", "price": 200000, "qty": 15},
    {"name": "Máy chiếu Epson 4K", "price": 35000000, "qty": 1},
    {"name": "Ghế công thái học", "price": 5000000, "qty": 6},
    {"name": "Webcam Logitech C930", "price": 2500000, "qty": 2},
    {"name": "Tủ mạng Rack 42U", "price": 15000000, "qty": 2},
]

# Ngưỡng cho đơn hàng VIP
VIP_THRESHOLD = 20000000

total_revenue = 0
vip_orders = []

print("-" * 60)
print(f"{'STT':<4} {'Tên Sản Phẩm':<30} {'Tổng Giá Trị':>15}  {'Xếp Loại'}")
print("-" * 60)

for index, item in enumerate(data, 1):
    order_value = item["price"] * item["qty"]
    total_revenue += order_value
    
    is_vip = order_value >= VIP_THRESHOLD
    rank = "VIP" if is_vip else "Phổ thông"
    
    if is_vip:
        vip_orders.append({
            "index": index,
            "name": item["name"],
            "value": order_value
        })
    
    # In ra từng dòng để kiểm tra (tùy chọn)
    print(f"{index:<4} {item['name']:<30} {order_value:>15,.0f}  {rank}")

print("-" * 60)
print(f"BÁO CÁO CỐI NGÀY")
print(f"Tổng doanh thu: {total_revenue:,.0f} VNĐ")
print("-" * 60)
print("DANH SÁCH ĐƠN HÀNG VIP (>= 20,000,000 VNĐ):")
for vip in vip_orders:
    print(f"- [Đơn #{vip['index']}] {vip['name']}: {vip['value']:,.0f} VNĐ")
