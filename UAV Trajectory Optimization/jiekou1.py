from flask import Flask, request, jsonify
import pymysql
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 数据库配置
DB_CONFIG = {
    'host': '192.168.50.25',
    'port': 33306,
    'user': 'root',
    'password': 'oneapimmysql',
    'database': 'demo',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """创建数据库连接"""
    return pymysql.connect(**DB_CONFIG)

@app.route('/canshu', methods=['POST'])
def canshu():
    try:
        # 优先从URL参数获取
        a = request.args.get('a')
        b = request.args.get('b')
        n = request.args.get('n')
        chatid = request.args.get('chatid') 
        
        # 如果URL参数中没有，再尝试其他来源
        if any(param is None for param in [a, b, n]):
            if request.is_json:
                data = request.get_json()
                a = a or data.get('a')
                b = b or data.get('b')
                n = n or data.get('n')
            else:
                a = a or request.form.get('a')
                b = b or request.form.get('b')
                n = n or request.form.get('n')

        # 验证所有必需参数是否存在
        if any(param is None for param in [a, b, n]):
            return jsonify({
                "success": False,
                "message": "Missing required parameters. 'a', 'b', and 'n' are required.",
                "received_data": {
                    "args": dict(request.args),
                    "form": dict(request.form),
                    "json": request.get_json() if request.is_json else None
                }
            }), 400

        try:
            # 参数类型转换
            a_value = float(a)
            b_value = float(b)
            n_value = int(n)
        except ValueError as e:
            return jsonify({
                "success": False,
                "message": f"Invalid parameter format: {str(e)}",
                "received_values": {
                    "a": a,
                    "b": b,
                    "n": n
                }
            }), 400

        # 打印接收到的参数，用于调试
        print(f"Received parameters: a={a_value}, b={b_value}, n={n_value}")

        # 创建数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # 执行插入操作
            sql = "INSERT INTO canshu2 (a, b, n,chatid) VALUES (%s, %s, %s,%s)"
            cursor.execute(sql, (a_value, b_value, n_value,chatid))
            
            # 提交事务
            conn.commit()

            return jsonify({
                "success": True,
                "message": "Parameters saved successfully",
                "data": {
                    "a": a_value,
                    "b": b_value,
                    "n": n_value
                }
            })

        except pymysql.Error as e:
            # 数据库错误处理
            conn.rollback()
            return jsonify({
                "success": False,
                "message": f"Database error: {str(e)}"
            }), 500

        finally:
            # 确保关闭连接
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error_type": str(type(e))
        }), 500

@app.route('/canshu3', methods=['POST'])
def canshu3():
    try:
        # 优先从URL参数获取
        a = request.args.get('a')
        b = request.args.get('b')
        n = request.args.get('n')
        w = request.args.get('w')
        chatid = request.args.get('chatid') 
        
        # 如果URL参数中没有，再尝试其他来源
        if any(param is None for param in [a, b, n]):
            if request.is_json:
                data = request.get_json()
                a = a or data.get('a')
                b = b or data.get('b')
                n = n or data.get('n')
                w = w or data.get('w')
            else:
                a = a or request.form.get('a')
                b = b or request.form.get('b')
                n = n or request.form.get('n')
                w = n or request.form.get('w')

        # 验证所有必需参数是否存在
        if any(param is None for param in [a, b,w, n]):
            return jsonify({
                "success": False,
                "message": "Missing required parameters. 'a', 'b', and 'n' are required.",
                "received_data": {
                    "args": dict(request.args),
                    "form": dict(request.form),
                    "json": request.get_json() if request.is_json else None
                }
            }), 400

        try:
            # 参数类型转换
            a_value = float(a)
            b_value = float(b)
            w_value = float(w)
            n_value = int(n)
        except ValueError as e:
            return jsonify({
                "success": False,
                "message": f"Invalid parameter format: {str(e)}",
                "received_values": {
                    "a": a,
                    "b": b,
                    "w": w,
                    "n": n
                }
            }), 400

        # 打印接收到的参数，用于调试
        print(f"Received parameters: a={a_value}, b={b_value}, n={n_value},w={w_value}")

        # 创建数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # 执行插入操作
            sql = "INSERT INTO pso (a, b, n,w,chatid) VALUES (%s, %s,%s, %s,%s)"
            cursor.execute(sql, (a_value, b_value, n_value,w_value,chatid))
            # 提交事务
            conn.commit()

            return jsonify({
                "success": True,
                "message": "Parameters saved successfully",
                "data": {
                    "a": a_value,
                    "b": b_value,
                    "w": w_value,
                    "n": n_value
                }
            })

        except pymysql.Error as e:
            # 数据库错误处理
            conn.rollback()
            return jsonify({
                "success": False,
                "message": f"Database error: {str(e)}"
            }), 500

        finally:
            # 确保关闭连接
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error_type": str(type(e))
        }), 500


@app.route('/canshu2', methods=['POST'])
def canshu2():
    try:
        # 优先从URL参数获取
        n = request.args.get('n')
        chatid = request.args.get('chatid')
        y = request.args.get('y')
        
        if not n or not chatid or not y:
            return jsonify({
                "success": False,
                "message": "Missing required parameters: n, chatid, or y"
            }), 400

        try:
            # 参数类型转换
            n_value = int(n) - 1
        except ValueError as e:
            return jsonify({
                "success": False,
                "message": f"Invalid parameter format: {str(e)}",
                "received_values": {
                    "n": n,
                    "y": y
                }
            }), 400

        # 创建数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # 执行更新操作
            sql = "UPDATE canshu2 SET y = %s WHERE chatid = %s AND n = %s"
            cursor.execute(sql, (y, chatid, n_value))
            
            # 检查是否有行被更新
            if cursor.rowcount == 0:
                return jsonify({
                    "success": False,
                    "message": "No matching record found to update"
                }), 404

            # 提交事务
            conn.commit()

            return jsonify({
                "success": True,
                "message": "Parameters updated successfully",
                "data": {
                    "chatid": chatid,
                    "n": n_value,
                    "y": y
                }
            })

        except pymysql.Error as e:
            # 数据库错误处理
            conn.rollback()
            return jsonify({
                "success": False,
                "message": f"Database error: {str(e)}"
            }), 500

        finally:
            # 确保关闭连接
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error_type": str(type(e))
        }), 500
@app.route('/canshu4', methods=['POST'])
def canshu4():
    try:
        # 优先从URL参数获取
        n = request.args.get('n')
        chatid = request.args.get('chatid')
        y = request.args.get('y')
        
        if not n or not chatid or not y:
            return jsonify({
                "success": False,
                "message": "Missing required parameters: n, chatid, or y"
            }), 400

        try:
            # 参数类型转换
            n_value = int(n) - 1
        except ValueError as e:
            return jsonify({
                "success": False,
                "message": f"Invalid parameter format: {str(e)}",
                "received_values": {
                    "n": n,
                    "y": y
                }
            }), 400

        # 创建数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # 执行更新操作
            sql = "UPDATE pso SET y = %s WHERE chatid = %s AND n = %s"
            cursor.execute(sql, (y, chatid, n_value))
            
            # 检查是否有行被更新
            if cursor.rowcount == 0:
                return jsonify({
                    "success": False,
                    "message": "No matching record found to update"
                }), 404

            # 提交事务
            conn.commit()

            return jsonify({
                "success": True,
                "message": "Parameters updated successfully",
                "data": {
                    "chatid": chatid,
                    "n": n_value,
                    "y": y
                }
            })

        except pymysql.Error as e:
            # 数据库错误处理
            conn.rollback()
            return jsonify({
                "success": False,
                "message": f"Database error: {str(e)}"
            }), 500

        finally:
            # 确保关闭连接
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error_type": str(type(e))
        }), 500

def conmysql(self, n):
    config = {
        'user': 'root',
        'password': 'zyz981129',
        'host': '127.0.0.1',
        'database': 'demo',
    }
    # 创建连接
    conn = pymysql.connect(**config)
    # 创建游标
    cursor = conn.cursor()
    # 执行SQL查询，按id降序排序
    cursor.execute('SELECT a FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    # 获取所有结果
    results = cursor.fetchall()
    for row in results:
        a = row[0]
    cursor.execute('SELECT b FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    results1 = cursor.fetchall()
    # 打印结果
    for row in results1:
        b = row[0]
    print(a)
    print(b)
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return a, b
@app.route('/select', methods=['POST'])
def select_by_chatid():
    try:
        # 从请求中获取 chatid
        chatid = request.args.get('chatid')
        
        if chatid is None:
            return jsonify({
                "success": False,
                "message": "Missing required parameter: 'chatid'."
            }), 400

        # 创建数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # 查询最新的50条记录
            sql = "SELECT * FROM canshu2 WHERE chatid=%s ORDER BY id DESC LIMIT 4"
            cursor.execute(sql, (chatid,))
            results = cursor.fetchall()

            # 将结果转换为字典列表
            records = []
            for row in results:
                record = {
                    "id": row[3],
                    "a": row[0],
                    "b": row[1],
                    "n": row[2],
                    "chatid": row[4],
                    "y": row[5]
                }
                records.append(record)

            return jsonify({
                "success": True,
                "data": records
            })

        except pymysql.Error as e:
            return jsonify({
                "success": False,
                "message": f"Database error: {str(e)}"
            }), 500

        finally:
            # 确保关闭连接
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error_type": str(type(e))
        }), 500
@app.route('/select2', methods=['POST'])
def select_by_chatid_pso():
    try:
        # 从请求中获取 chatid
        chatid = request.args.get('chatid')
        
        if chatid is None:
            return jsonify({
                "success": False,
                "message": "Missing required parameter: 'chatid'."
            }), 400

        # 创建数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # 查询最新的50条记录
            sql = "SELECT * FROM pso WHERE chatid=%s ORDER BY id DESC LIMIT 4"
            cursor.execute(sql, (chatid,))
            results = cursor.fetchall()

            # 将结果转换为字典列表
            records = []
            for row in results:
                record = {
                    "id": row[4],
                    "a": row[0],
                    "b": row[1],
                    "n": row[3],
                    "w": row[2],
                    "chatid": row[5],
                    "y": row[6]
                }
                records.append(record)

            return jsonify({
                "success": True,
                "data": records
            })

        except pymysql.Error as e:
            return jsonify({
                "success": False,
                "message": f"Database error: {str(e)}"
            }), 500

        finally:
            # 确保关闭连接
            cursor.close()
            conn.close()

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error_type": str(type(e))
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
