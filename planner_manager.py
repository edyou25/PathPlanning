import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import time

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PlannerType(Enum):
    """支持的规划器类型"""
    # 基于搜索的2D规划器
    A_STAR_2D = "a_star_2d"
    BIDIRECTIONAL_A_STAR_2D = "bidirectional_a_star_2d"
    ARA_STAR_2D = "ara_star_2d"
    
    # 基于搜索的3D规划器
    A_STAR_3D = "a_star_3d"
    BIDIRECTIONAL_A_STAR_3D = "bidirectional_a_star_3d"
    D_STAR_LITE_3D = "d_star_lite_3d"
    ANYTIME_D_STAR_3D = "anytime_d_star_3d"
    LRT_A_STAR_3D = "lrt_a_star_3d"
    RTA_A_STAR_3D = "rta_a_star_3d"
    LP_A_STAR_3D = "lp_a_star_3d"
    
    # 基于采样的2D规划器
    RRT_2D = "rrt_2d"
    RRT_STAR_2D = "rrt_star_2d"
    RRT_STAR_SMART_2D = "rrt_star_smart_2d"
    DUBINS_RRT_STAR_2D = "dubins_rrt_star_2d"
    FMT_2D = "fmt_2d"
    
    # 基于采样的3D规划器
    RRT_3D = "rrt_3d"
    RRT_CONNECT_3D = "rrt_connect_3d"
    RRT_STAR_3D = "rrt_star_3d"
    ABIT_STAR_3D = "abit_star_3d"
    FMT_STAR_3D = "fmt_star_3d"
    
    # 曲线生成器
    BEZIER_PATH = "bezier_path"
    BSPLINE_CURVE = "bspline_curve"
    CUBIC_SPLINE = "cubic_spline"
    DUBINS_PATH = "dubins_path"
    REEDS_SHEPP = "reeds_shepp"
    QUARTIC_POLYNOMIAL = "quartic_polynomial"
    QUINTIC_POLYNOMIAL = "quintic_polynomial"

class PlannerDimension(Enum):
    """规划器维度"""
    TWO_D = "2D"
    THREE_D = "3D"
    CURVE = "CURVE"

class PlannerCategory(Enum):
    """规划器分类"""
    SEARCH_BASED = "search_based"
    SAMPLING_BASED = "sampling_based"
    CURVE_GENERATOR = "curve_generator"

class PathPlannerManager:
    """路径规划管理器 - 统一调用接口"""
    
    def __init__(self):
        self.current_planner = None
        self.current_planner_type = None
        self.planning_results = {}
        self.planner_configs = self._load_default_configs()
        
    def _load_default_configs(self) -> Dict[PlannerType, Dict[str, Any]]:
        """加载默认配置"""
        configs = {}
        
        # 2D搜索算法配置
        configs[PlannerType.A_STAR_2D] = {
            'heuristic_type': 'manhattan',
            'resolution': 1.0
        }
        
        # 3D搜索算法配置
        configs[PlannerType.A_STAR_3D] = {
            'resolution': 0.5,
            'weight': 1.0
        }
        
        # 2D采样算法配置
        configs[PlannerType.RRT_STAR_2D] = {
            'step_len': 0.5,
            'goal_sample_rate': 0.1,
            'search_radius': 1.0,
            'iter_max': 4000
        }
        
        # 3D采样算法配置
        configs[PlannerType.RRT_CONNECT_3D] = {
            'max_iter': 1000,
            'step_size': 0.5,
            'goal_bias': 0.1
        }
        
        # 曲线生成器配置
        configs[PlannerType.CUBIC_SPLINE] = {
            'resolution': 0.1
        }
        
        return configs
    
    def set_planner_config(self, planner_type: PlannerType, config: Dict[str, Any]):
        """设置规划器配置"""
        self.planner_configs[planner_type] = config
    
    def get_planner_info(self, planner_type: PlannerType) -> Dict[str, str]:
        """获取规划器信息"""
        info_map = {
            PlannerType.A_STAR_2D: {
                'name': 'A* 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SEARCH_BASED.value,
                'description': '2D A*搜索算法'
            },
            PlannerType.A_STAR_3D: {
                'name': 'A* 3D',
                'dimension': PlannerDimension.THREE_D.value,
                'category': PlannerCategory.SEARCH_BASED.value,
                'description': '3D A*搜索算法'
            },
            PlannerType.RRT_STAR_2D: {
                'name': 'RRT* 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '2D RRT*采样算法'
            },
            PlannerType.RRT_CONNECT_3D: {
                'name': 'RRT-Connect 3D',
                'dimension': PlannerDimension.THREE_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '3D RRT-Connect双向采样算法'
            },
            PlannerType.CUBIC_SPLINE: {
                'name': 'Cubic Spline',
                'dimension': PlannerDimension.CURVE.value,
                'category': PlannerCategory.CURVE_GENERATOR.value,
                'description': '三次样条曲线生成器'
            }
        }
        return info_map.get(planner_type, {'name': 'Unknown', 'dimension': 'Unknown', 'category': 'Unknown', 'description': 'Unknown planner'})
    
    def plan_path(self, 
                  start: Tuple[float, ...], 
                  goal: Tuple[float, ...], 
                  planner_type: PlannerType,
                  config_override: Optional[Dict] = None) -> Optional[List[Tuple[float, ...]]]:
        """
        统一的路径规划接口
        
        Args:
            start: 起始点坐标
            goal: 目标点坐标
            planner_type: 规划器类型
            config_override: 配置覆盖参数
            
        Returns:
            规划的路径点列表或None
        """
        # 合并配置
        config = self.planner_configs.get(planner_type, {}).copy()
        if config_override:
            config.update(config_override)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 根据规划器类型调用对应的规划算法
            path = self._call_planner(start, goal, planner_type, config)
            
            # 记录结果
            planning_time = time.time() - start_time
            self.planning_results[planner_type] = {
                'path': path,
                'planning_time': planning_time,
                'success': path is not None,
                'path_length': self._calculate_path_length(path) if path else 0
            }
            
            print(f"规划器 {planner_type.value} 完成，用时: {planning_time:.3f}s")
            if path:
                print(f"路径长度: {len(path)} 点，总距离: {self.planning_results[planner_type]['path_length']:.3f}")
            else:
                print("路径规划失败")
            
            return path
            
        except Exception as e:
            import traceback
            print(f"规划器 {planner_type.value} 执行失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("详细错误堆栈:")
            print(traceback.format_exc())
            
            # 记录失败结果
            planning_time = time.time() - start_time
            self.planning_results[planner_type] = {
                'path': None,
                'planning_time': planning_time,
                'success': False,
                'path_length': 0,
                'error': str(e),
                'error_type': type(e).__name__
            }
            return None
    
    def _call_planner(self, start, goal, planner_type, config):
        """调用具体的规划器"""
        if planner_type == PlannerType.A_STAR_2D:
            return self._call_astar_2d(start, goal, config)
        elif planner_type == PlannerType.A_STAR_3D:
            return self._call_astar_3d(start, goal, config)
        elif planner_type == PlannerType.RRT_STAR_2D:
            return self._call_rrt_star_2d(start, goal, config)
        elif planner_type == PlannerType.RRT_CONNECT_3D:
            return self._call_rrt_connect_3d(start, goal, config)
        elif planner_type == PlannerType.CUBIC_SPLINE:
            return self._call_cubic_spline(start, goal, config)
        else:
            raise NotImplementedError(f"规划器 {planner_type.value} 尚未实现")
    
    def _call_astar_2d(self, start, goal, config):
        """调用2D A*算法"""
        # 将浮点坐标转换为整数坐标（A*算法期望整数格网坐标）
        start_int = (int(round(start[0])), int(round(start[1])))
        goal_int = (int(round(goal[0])), int(round(goal[1])))
        
        print(f"原始坐标 - 起始点: {start}, 目标点: {goal}")
        print(f"转换坐标 - 起始点: {start_int}, 目标点: {goal_int}")
        
        # 导入A*算法
        from Search_based_Planning.Search_2D.Astar import AStar
        
        # 尝试不同的构造函数参数组合
        try:
            planner = AStar(start_int, goal_int, config.get('heuristic_type', 'manhattan'))
        except TypeError:
            # 如果参数不匹配，尝试其他方式
            try:
                planner = AStar(s_start=start_int, s_goal=goal_int, heuristic_type=config.get('heuristic_type', 'manhattan'))
            except TypeError:
                # 如果还是不行，尝试最简单的参数
                planner = AStar(start_int, goal_int)
        
        path, _ = planner.searching()
        
        # 将整数路径坐标转换回浮点数坐标
        if path:
            path = [(float(p[0]), float(p[1])) for p in path]
        
        return path
    
    def _call_astar_3d(self, start, goal, config):
        """调用3D A*算法"""
        # 添加必要的路径到sys.path
        search_3d_path = os.path.join(os.path.dirname(__file__), 'Search_based_Planning', 'Search_3D')
        if search_3d_path not in sys.path:
            sys.path.append(search_3d_path)
        
        # 确保搜索路径正确
        search_base_path = os.path.join(os.path.dirname(__file__), 'Search_based_Planning')
        if search_base_path not in sys.path:
            sys.path.append(search_base_path)
        
        # 导入3D A*算法
        from Search_based_Planning.Search_3D.Astar3D import Weighted_A_star
        
        planner = Weighted_A_star(resolution=config.get('resolution', 0.5))
        
        # 设置起始点和目标点
        planner.env.start = np.array(start)
        planner.env.goal = np.array(goal)
        planner.start, planner.goal = tuple(start), tuple(goal)
        planner.x0, planner.xt = planner.start, planner.goal
        
        success = planner.run()
        if success and planner.Path:
            return [(float(p[0]), float(p[1]), float(p[2])) for p in planner.Path]
        return None
    
    def _call_rrt_star_2d(self, start, goal, config):
        """调用2D RRT*算法"""
        from Sampling_based_Planning.rrt_2D.rrt_star import RrtStar
        
        planner = RrtStar(
            x_start=start,
            x_goal=goal,
            step_len=config.get('step_len', 10),
            goal_sample_rate=config.get('goal_sample_rate', 0.1),
            search_radius=config.get('search_radius', 1.0),
            iter_max=config.get('iter_max', 500)
        )
        
        path = planner.planning()
        return path
    
    def _call_rrt_connect_3d(self, start, goal, config):
        """调用3D RRT-Connect算法"""
        from Sampling_based_Planning.rrt_3D.rrt_connect3D import rrt_connect
        
        planner = rrt_connect()
        planner.maxiter = config.get('max_iter', 1000)
        
        result = planner.RRT_CONNECT_PLANNER(start, goal)
        if planner.done and planner.Path:
            return [(float(p[0]), float(p[1]), float(p[2])) for p in planner.Path]
        return None
    
    def _call_cubic_spline(self, start, goal, config):
        """调用三次样条曲线生成器"""
        from CurvesGenerator.cubic_spline import Spline2D
        
        # 创建简单的waypoints
        waypoints = [start, goal]
        
        # 如果只有两个点，添加中间点
        if len(waypoints) == 2:
            mid_x = (start[0] + goal[0]) / 2
            mid_y = (start[1] + goal[1]) / 2
            waypoints = [start, (mid_x, mid_y), goal]
        
        x_coords = [p[0] for p in waypoints]
        y_coords = [p[1] for p in waypoints]
        
        spline = Spline2D(x_coords, y_coords)
        
        # 生成路径点
        resolution = config.get('resolution', 0.1)
        t_values = np.arange(0, len(x_coords), resolution)
        path = []
        
        for t in t_values:
            if t < len(x_coords) - 1:
                x, y = spline.calc_position(t)
                path.append((float(x), float(y)))
        
        return path
    
    def _calculate_path_length(self, path: List[Tuple[float, ...]]) -> float:
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            total_length += dist
        
        return total_length
    
    def compare_planners(self, 
                        start: Tuple[float, ...], 
                        goal: Tuple[float, ...], 
                        planner_types: List[PlannerType]) -> Dict[PlannerType, Dict]:
        """比较多个规划器的性能"""
        results = {}
        
        print("开始规划器性能比较...")
        print(f"起始点: {start}")
        print(f"目标点: {goal}")
        print(f"比较规划器: {[p.value for p in planner_types]}")
        print("-" * 50)
        
        for planner_type in planner_types:
            print(f"测试规划器: {planner_type.value}")
            path = self.plan_path(start, goal, planner_type)
            results[planner_type] = self.planning_results.get(planner_type, {})
            print("-" * 30)
        
        # 显示比较结果
        self._display_comparison_results(results)
        return results
    
    def _display_comparison_results(self, results: Dict[PlannerType, Dict]):
        """显示比较结果"""
        print("\n规划器性能比较结果:")
        print("=" * 80)
        print(f"{'规划器':<20} {'成功':<6} {'用时(s)':<8} {'路径点数':<8} {'路径长度':<10}")
        print("-" * 80)
        
        for planner_type, result in results.items():
            success = "✓" if result.get('success', False) else "✗"
            time_str = f"{result.get('planning_time', 0):.3f}"
            points = len(result.get('path', [])) if result.get('path') else 0
            length = f"{result.get('path_length', 0):.3f}"
            
            print(f"{planner_type.value:<20} {success:<6} {time_str:<8} {points:<8} {length:<10}")
    
    def visualize_environment_2d(self, planner_type: PlannerType, ax=None, title: str = "2D Environment"):
        """可视化2D环境障碍物"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            show_plot = True
        else:
            show_plot = False
            
        if planner_type in [PlannerType.A_STAR_2D]:
            # Search_2D环境
            from Search_based_Planning.Search_2D.env import Env
            env = Env()
            
            # 绘制边界
            ax.add_patch(patches.Rectangle((0, 0), env.x_range, env.y_range, 
                                         linewidth=2, edgecolor='black', facecolor='none'))
            
            # 绘制障碍物点
            if hasattr(env, 'obs') and env.obs:
                obs_x = [obs[0] for obs in env.obs]
                obs_y = [obs[1] for obs in env.obs]
                ax.scatter(obs_x, obs_y, c='red', s=5, alpha=0.6, label='Obstacles')
            
            ax.set_xlim(-1, env.x_range + 1)
            ax.set_ylim(-1, env.y_range + 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
            
        elif planner_type in [PlannerType.RRT_STAR_2D, PlannerType.RRT_2D]:
            # Sampling_2D环境
            from Sampling_based_Planning.rrt_2D.env import Env
            env = Env()
            
            # 绘制边界障碍物
            for boundary in env.obs_boundary:
                x, y, w, h = boundary
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor='black', facecolor='gray', alpha=0.8)
                ax.add_patch(rect)
            
            # 绘制矩形障碍物
            for rectangle in env.obs_rectangle:
                x, y, w, h = rectangle
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor='red', facecolor='red', alpha=0.6)
                ax.add_patch(rect)
            
            # 绘制圆形障碍物
            for circle in env.obs_circle:
                x, y, r = circle
                circle_patch = patches.Circle((x, y), r, linewidth=1, 
                                            edgecolor='blue', facecolor='blue', alpha=0.6)
                ax.add_patch(circle_patch)
            
            ax.set_xlim(env.x_range[0] - 2, env.x_range[1] + 2)
            ax.set_ylim(env.y_range[0] - 2, env.y_range[1] + 2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # 添加图例
            boundary_patch = patches.Patch(color='gray', alpha=0.8, label='Boundary')
            rect_patch = patches.Patch(color='red', alpha=0.6, label='Rectangle Obstacles')
            circle_patch = patches.Patch(color='blue', alpha=0.6, label='Circle Obstacles')
            ax.legend(handles=[boundary_patch, rect_patch, circle_patch])
        
        if show_plot:
            plt.tight_layout()
            plt.show()
            return plt.gcf(), ax
        
        return None, ax
    
    def visualize_environment_3d(self, planner_type: PlannerType, ax=None, title: str = "3D Environment"):
        """可视化3D环境障碍物"""
        if planner_type in [PlannerType.A_STAR_3D]:
            # Search_3D环境
            from Search_based_Planning.Search_3D.env3D import env
            env_3d = env()
            
        elif planner_type in [PlannerType.RRT_CONNECT_3D, PlannerType.RRT_3D]:
            # Sampling_3D环境
            from Sampling_based_Planning.rrt_3D.env3D import env
            env_3d = env()
        else:
            print(f"不支持的3D规划器类型: {planner_type}")
            return None, None
        
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            show_plot = True
        else:
            show_plot = False
        
        # 绘制工作空间边界
        boundary = env_3d.boundary
        xmin, ymin, zmin, xmax, ymax, zmax = boundary
        
        # 绘制边界框架
        # 底面
        ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], 'k-', alpha=0.3)
        ax.plot([xmin, xmax], [ymax, ymax], [zmin, zmin], 'k-', alpha=0.3)
        ax.plot([xmin, xmin], [ymin, ymax], [zmin, zmin], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], 'k-', alpha=0.3)
        
        # 顶面
        ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], 'k-', alpha=0.3)
        ax.plot([xmin, xmax], [ymax, ymax], [zmax, zmax], 'k-', alpha=0.3)
        ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], 'k-', alpha=0.3)
        
        # 垂直边
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], 'k-', alpha=0.3)
        ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], 'k-', alpha=0.3)
        
        # 绘制AABB障碍物（立方体）
        if hasattr(env_3d, 'blocks') and len(env_3d.blocks) > 0:
            for i, block in enumerate(env_3d.blocks):
                x1, y1, z1, x2, y2, z2 = block
                
                # 绘制立方体的12条边
                # 底面4条边
                ax.plot([x1, x2], [y1, y1], [z1, z1], 'r-', linewidth=2)
                ax.plot([x1, x2], [y2, y2], [z1, z1], 'r-', linewidth=2)
                ax.plot([x1, x1], [y1, y2], [z1, z1], 'r-', linewidth=2)
                ax.plot([x2, x2], [y1, y2], [z1, z1], 'r-', linewidth=2)
                
                # 顶面4条边
                ax.plot([x1, x2], [y1, y1], [z2, z2], 'r-', linewidth=2)
                ax.plot([x1, x2], [y2, y2], [z2, z2], 'r-', linewidth=2)
                ax.plot([x1, x1], [y1, y2], [z2, z2], 'r-', linewidth=2)
                ax.plot([x2, x2], [y1, y2], [z2, z2], 'r-', linewidth=2)
                
                # 垂直4条边
                ax.plot([x1, x1], [y1, y1], [z1, z2], 'r-', linewidth=2)
                ax.plot([x2, x2], [y1, y1], [z1, z2], 'r-', linewidth=2)
                ax.plot([x1, x1], [y2, y2], [z1, z2], 'r-', linewidth=2)
                ax.plot([x2, x2], [y2, y2], [z1, z2], 'r-', linewidth=2)
        
        # 绘制球体障碍物
        if hasattr(env_3d, 'balls') and len(env_3d.balls) > 0:
            for ball in env_3d.balls:
                cx, cy, cz, r = ball
                
                # 创建球面
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = cx + r * np.outer(np.cos(u), np.sin(v))
                y_sphere = cy + r * np.outer(np.sin(u), np.sin(v))
                z_sphere = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')
        
        # 绘制起始点和目标点
        if hasattr(env_3d, 'start'):
            ax.scatter(env_3d.start[0], env_3d.start[1], env_3d.start[2], 
                      c='green', s=100, label='Start', marker='o')
        
        if hasattr(env_3d, 'goal'):
            ax.scatter(env_3d.goal[0], env_3d.goal[1], env_3d.goal[2], 
                      c='red', s=100, label='Goal', marker='s')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title}")
        ax.legend()
        
        # 设置相等的坐标轴比例
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
        mid_x = (xmax + xmin) * 0.5
        mid_y = (ymax + ymin) * 0.5
        mid_z = (zmax + zmin) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
            return plt.gcf(), ax
        
        return None, ax
    
    def visualize_path_with_environment(self, 
                                      path: List[Tuple[float, ...]], 
                                      start: Tuple[float, ...], 
                                      goal: Tuple[float, ...],
                                      planner_type: PlannerType,
                                      ax=None,
                                      title: str = "Path Planning Result"):
        """可视化路径和环境"""
        if not path:
            print("没有路径可以可视化")
            return None, None
        
        if len(path[0]) == 2:  # 2D路径
            _, ax = self.visualize_environment_2d(planner_type, ax, title)
            
            # 绘制路径
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            ax.plot(x_coords, y_coords, 'g-', linewidth=3, label='Path', alpha=0.8)
            
            # 绘制起始点和目标点
            ax.plot(start[0], start[1], 'go', markersize=12, label='Start', markeredgecolor='black')
            ax.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal', markeredgecolor='black')
            
            ax.legend()
            
        elif len(path[0]) == 3:  # 3D路径
            _, ax = self.visualize_environment_3d(planner_type, ax, title)
            
            # 绘制路径
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            z_coords = [p[2] for p in path]
            ax.plot(x_coords, y_coords, z_coords, 'g-', linewidth=3, label='Path', alpha=0.8)
            
            # 重新绘制起始点和目标点（覆盖环境中的点）
            ax.scatter(start[0], start[1], start[2], c='green', s=150, label='Start', 
                      marker='o', edgecolors='black', linewidth=2)
            ax.scatter(goal[0], goal[1], goal[2], c='red', s=150, label='Goal', 
                      marker='s', edgecolors='black', linewidth=2)
            
            ax.legend()
        
        return None, ax
    
    def get_available_planners(self) -> Dict[PlannerCategory, List[PlannerType]]:
        """获取可用的规划器列表，按类别分组"""
        planners = {
            PlannerCategory.SEARCH_BASED: [
                PlannerType.A_STAR_2D,
                PlannerType.A_STAR_3D,
                PlannerType.BIDIRECTIONAL_A_STAR_2D,
                PlannerType.BIDIRECTIONAL_A_STAR_3D,
            ],
            PlannerCategory.SAMPLING_BASED: [
                PlannerType.RRT_STAR_2D,
                PlannerType.RRT_CONNECT_3D,
                PlannerType.RRT_STAR_SMART_2D,
                PlannerType.ABIT_STAR_3D,
            ],
            PlannerCategory.CURVE_GENERATOR: [
                PlannerType.CUBIC_SPLINE,
                PlannerType.BEZIER_PATH,
                PlannerType.DUBINS_PATH,
            ]
        }
        return planners
    
    def get_planning_results(self) -> Dict[PlannerType, Dict]:
        """获取所有规划结果"""
        return self.planning_results.copy()

# 使用示例
if __name__ == "__main__":
    # 创建路径规划管理器
    manager = PathPlannerManager()
    
    print("=== 统一环境可视化演示 ===")
    
    # 创建一个大的figure来显示所有结果
    fig = plt.figure(figsize=(20, 12))
    
    # 2D路径规划示例
    start_2d = (5.0, 5.0)
    goal_2d = (45.0, 25.0)
    
    print("\n--- 2D环境可视化和路径规划 ---")
    
    # 子图1: Search_2D环境
    ax1 = fig.add_subplot(2, 3, 1)
    manager.visualize_environment_2d(PlannerType.A_STAR_2D, ax1, "Search_2D Environment")
    
    # 子图2: Sampling_2D环境
    ax2 = fig.add_subplot(2, 3, 2)
    manager.visualize_environment_2d(PlannerType.RRT_STAR_2D, ax2, "Sampling_2D Environment")
    
    # 使用A* 2D规划器
    path_astar_2d = manager.plan_path(start_2d, goal_2d, PlannerType.A_STAR_2D)
    if path_astar_2d:
        print(f"A* 2D规划成功，路径包含 {len(path_astar_2d)} 个点")
        # 子图3: A* 2D路径规划结果
        ax3 = fig.add_subplot(2, 3, 3)
        manager.visualize_path_with_environment(path_astar_2d, start_2d, goal_2d, 
                                              PlannerType.A_STAR_2D, ax3, "A* 2D Path Planning")
    
    # 使用RRT* 2D规划器
    path_rrt_star_2d = manager.plan_path(start_2d, goal_2d, PlannerType.RRT_STAR_2D)
    if path_rrt_star_2d:
        print(f"RRT* 2D规划成功，路径包含 {len(path_rrt_star_2d)} 个点")
        # 子图4: RRT* 2D路径规划结果
        ax4 = fig.add_subplot(2, 3, 4)
        manager.visualize_path_with_environment(path_rrt_star_2d, start_2d, goal_2d, 
                                              PlannerType.RRT_STAR_2D, ax4, "RRT* 2D Path Planning")
    plt.tight_layout()
    plt.show()
    exit(0)  # 退出以避免继续执行3D部分
    
    # 3D路径规划示例
    start_3d = (1.0, 1.0, 1.0)
    goal_3d = (18.0, 18.0, 4.0)
    
    print("\n--- 3D环境可视化和路径规划 ---")
    
    # 子图5: 3D环境
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    manager.visualize_environment_3d(PlannerType.A_STAR_3D, ax5, "3D Environment")
    
    # 使用A* 3D规划器
    path_astar_3d = manager.plan_path(start_3d, goal_3d, PlannerType.A_STAR_3D)
    if path_astar_3d:
        print(f"A* 3D规划成功，路径包含 {len(path_astar_3d)} 个点")
        # 子图6: A* 3D路径规划结果
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        manager.visualize_path_with_environment(path_astar_3d, start_3d, goal_3d, 
                                              PlannerType.A_STAR_3D, ax6, "A* 3D Path Planning")
    
    # 调整子图间距并显示
    plt.tight_layout()
    plt.show()
    
    # 演示规划器性能比较
    print("\n=== 规划器性能比较 ===")
    
    # 2D规划器比较
    print("\n--- 2D规划器比较 ---")
    planners_2d = [PlannerType.A_STAR_2D, PlannerType.CUBIC_SPLINE]
    comparison_2d = manager.compare_planners(start_2d, goal_2d, planners_2d)
    
    # 3D规划器比较  
    print("\n--- 3D规划器比较 ---")
    planners_3d = [PlannerType.A_STAR_3D]
    comparison_3d = manager.compare_planners(start_3d, goal_3d, planners_3d)
    
    # 显示可用的规划器
    print("\n=== 可用规划器 ===")
    available_planners = manager.get_available_planners()
    for category, planners in available_planners.items():
        print(f"{category.value}:")
        for planner in planners:
            info = manager.get_planner_info(planner)
            print(f"  - {info['name']} ({planner.value}): {info['description']}")
    
    print("\n=== 环境可视化演示完成 ===")
    print("现在可以直接使用各个算法原有的环境定义进行规划和可视化！")