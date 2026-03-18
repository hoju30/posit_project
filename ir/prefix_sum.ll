; ModuleID = '/home/hoju/test/posit_test/src/ir_kernels/prefix_sum.cpp'
source_filename = "/home/hoju/test/posit_test/src/ir_kernels/prefix_sum.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (double (double*, i64)* @kernel_prefix_sum_total to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @kernel_prefix_sum_total(double* nocapture noundef readonly %0, i64 noundef %1) #0 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %27, label %4

4:                                                ; preds = %2
  %5 = and i64 %1, 3
  %6 = icmp ult i64 %1, 4
  br i1 %6, label %9, label %7

7:                                                ; preds = %4
  %8 = and i64 %1, -4
  br label %29

9:                                                ; preds = %29, %4
  %10 = phi double [ undef, %4 ], [ %52, %29 ]
  %11 = phi i64 [ 0, %4 ], [ %53, %29 ]
  %12 = phi double [ 0.000000e+00, %4 ], [ %52, %29 ]
  %13 = phi double [ 0.000000e+00, %4 ], [ %51, %29 ]
  %14 = icmp eq i64 %5, 0
  br i1 %14, label %27, label %15

15:                                               ; preds = %9, %15
  %16 = phi i64 [ %24, %15 ], [ %11, %9 ]
  %17 = phi double [ %23, %15 ], [ %12, %9 ]
  %18 = phi double [ %22, %15 ], [ %13, %9 ]
  %19 = phi i64 [ %25, %15 ], [ 0, %9 ]
  %20 = getelementptr inbounds double, double* %0, i64 %16
  %21 = load double, double* %20, align 8, !tbaa !5
  %22 = fadd double %18, %21
  %23 = fadd double %17, %22
  %24 = add nuw i64 %16, 1
  %25 = add i64 %19, 1
  %26 = icmp eq i64 %25, %5
  br i1 %26, label %27, label %15, !llvm.loop !9

27:                                               ; preds = %9, %15, %2
  %28 = phi double [ 0.000000e+00, %2 ], [ %10, %9 ], [ %23, %15 ]
  ret double %28

29:                                               ; preds = %29, %7
  %30 = phi i64 [ 0, %7 ], [ %53, %29 ]
  %31 = phi double [ 0.000000e+00, %7 ], [ %52, %29 ]
  %32 = phi double [ 0.000000e+00, %7 ], [ %51, %29 ]
  %33 = phi i64 [ 0, %7 ], [ %54, %29 ]
  %34 = getelementptr inbounds double, double* %0, i64 %30
  %35 = load double, double* %34, align 8, !tbaa !5
  %36 = fadd double %32, %35
  %37 = fadd double %31, %36
  %38 = or i64 %30, 1
  %39 = getelementptr inbounds double, double* %0, i64 %38
  %40 = load double, double* %39, align 8, !tbaa !5
  %41 = fadd double %36, %40
  %42 = fadd double %37, %41
  %43 = or i64 %30, 2
  %44 = getelementptr inbounds double, double* %0, i64 %43
  %45 = load double, double* %44, align 8, !tbaa !5
  %46 = fadd double %41, %45
  %47 = fadd double %42, %46
  %48 = or i64 %30, 3
  %49 = getelementptr inbounds double, double* %0, i64 %48
  %50 = load double, double* %49, align 8, !tbaa !5
  %51 = fadd double %46, %50
  %52 = fadd double %47, %51
  %53 = add nuw i64 %30, 4
  %54 = add i64 %33, 4
  %55 = icmp eq i64 %54, %8
  br i1 %55, label %9, label %29, !llvm.loop !11
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 16.0.6 (23ubuntu4)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
